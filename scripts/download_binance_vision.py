#!/usr/bin/env python3
"""
Download Binance klines from data.binance.vision (Public Data) and convert to RD-Agent h5.

Supports all intervals including 1s (spot only for 1s).
Downloads one month/day at a time, processes it, then deletes the zip to save disk.

Usage:
    # Spot 1s klines → crypto_1s.h5 (default, for ETHUSDT scenario)
    python scripts/download_binance_public.py

    # Other timeframes / date ranges
    python scripts/download_binance_public.py --timeframe 1m
    python scripts/download_binance_public.py --start 2025-01-01 --end 2025-12-31
    python scripts/download_binance_public.py --type um --timeframe 1m  # futures
"""

import argparse
import logging
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL = "https://data.binance.vision"
OUTPUT_DIR = Path("git_ignore_folder/factor_implementation_source_data")
OUTPUT_FILE = "crypto_1s.h5"
HDF_KEY = "data"

# Binance kline CSV columns
_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
]
RDAGENT_COLS = ["$open", "$close", "$high", "$low", "$volume", "$factor"]

log = logging.getLogger("download_binance_public")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _symbol_to_instrument(symbol: str) -> str:
    return symbol.replace("/", "")


def _save_hdf(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(path, key=HDF_KEY, mode="w")
    log.info("Saved %s rows to %s", len(df), path)


def _to_rdagent_format(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    out = df.rename(columns={
        "open": "$open", "high": "$high", "low": "$low",
        "close": "$close", "volume": "$volume",
    })
    out["$factor"] = 1.0
    out.index.name = "datetime"
    out["instrument"] = instrument
    out = out.reset_index().set_index(["datetime", "instrument"]).sort_index()
    return out[RDAGENT_COLS]


def _build_urls(symbol: str, interval: str, trading_type: str,
                start: str, end: str) -> list[tuple[str, str]]:
    """Build (url, label) pairs for monthly + daily zip files."""
    urls = []
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    prefix = "data/spot" if trading_type == "spot" else f"data/futures/{trading_type}"

    # Monthly files
    current = start_dt.replace(day=1)
    while current <= end_dt:
        filename = f"{symbol.upper()}-{interval}-{current.strftime('%Y-%m')}.zip"
        url = f"{BASE_URL}/{prefix}/monthly/klines/{symbol.upper()}/{interval}/{filename}"
        urls.append((url, f"monthly_{current.strftime('%Y-%m')}"))
        current += pd.DateOffset(months=1)

    # Daily files (covers partial months and most recent data not yet in monthly)
    dates = pd.date_range(start_dt, end_dt, freq="D")
    for d in dates:
        filename = f"{symbol.upper()}-{interval}-{d.strftime('%Y-%m-%d')}.zip"
        url = f"{BASE_URL}/{prefix}/daily/klines/{symbol.upper()}/{interval}/{filename}"
        urls.append((url, f"daily_{d.strftime('%Y-%m-%d')}"))

    return urls


def _download_and_parse(url: str) -> pd.DataFrame | None:
    """Download a single zip, parse CSV into DataFrame, delete zip immediately."""
    try:
        resp = urllib.request.urlopen(url)
        data = resp.read()
    except urllib.error.HTTPError:
        log.warning("Not found (skipping): %s", url)
        return None

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "data.zip"
        zip_path.write_bytes(data)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return None
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f, header=None, names=_KLINE_COLUMNS)
        except zipfile.BadZipFile:
            log.warning("Bad zip: %s", url)
            return None
    # tmp dir is auto-deleted here, zip gone

    # Binance 1s klines use microseconds, others use milliseconds — auto-detect
    ts0 = int(df["open_time"].iloc[0])
    unit = "us" if ts0 > 1e14 else "ms"
    df["datetime"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    for col in df.columns:
        df[col] = df[col].astype(np.float64)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download Binance Public Data klines → RD-Agent h5",
    )
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--timeframe", default="1s")
    parser.add_argument("--type", choices=["spot", "um", "cm"], default="spot",
                        help="Market type (default: spot; 1s only available on spot)")
    parser.add_argument("--start", default="2025-10-31")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--output", help="Output .h5 path (default: auto)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.timeframe == "1s" and args.type != "spot":
        log.warning("1s klines only available on spot. Switching to spot.")
        args.type = "spot"

    instrument = _symbol_to_instrument(args.symbol)
    out_path = Path(args.output) if args.output else OUTPUT_DIR / OUTPUT_FILE

    urls = _build_urls(args.symbol, args.timeframe, args.type, args.start, args.end)
    log.info("Will process %d files (monthly + daily, duplicates auto-merged)", len(urls))

    import time
    t0 = time.time()
    frames = []
    for i, (url, label) in enumerate(urls, 1):
        df = _download_and_parse(url)
        if df is not None:
            frames.append(df)
            log.info("[%d/%d] %s: %d rows", i, len(urls), label, len(df))
        else:
            log.debug("[%d/%d] %s: skipped", i, len(urls), label)

    if not frames:
        log.error("No data downloaded. Check symbol/timeframe/date range/type.")
        sys.exit(1)

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Trim to exact date range
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df.loc[start_ts:end_ts]

    out = _to_rdagent_format(df, instrument)
    _save_hdf(out, out_path)

    elapsed = time.time() - t0
    log.info("Completed in %.0fs", elapsed)
    log.info("  Range: %s → %s", out.index.get_level_values("datetime").min(),
             out.index.get_level_values("datetime").max())
    log.info("  Rows:  %d", len(out))


if __name__ == "__main__":
    main()

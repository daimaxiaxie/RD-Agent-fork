#!/usr/bin/env python3
"""
Download Binance OHLCV data via CCXT and convert to RD-Agent daily_pv.h5 format.

Features:
- Downloads kline data from Binance at configurable timeframe (1m, 1h, 1d, etc.)
- Resamples to daily OHLCV with proper MultiIndex (datetime, instrument)
- Outputs HDF5 with key="data" matching RD-Agent's expected format
- Extensible: swap --symbol to download any pair (ETH/USDT, ETH/USDC, BTC/USDT, etc.)
- Automatic rate limiting and retry on Binance API errors
- Resumes partial downloads via per-chunk caching

Usage:
    # ETH/USDT 1m klines → daily_pv.h5 (default, recommended)
    python scripts/download_binance_data.py

    # Other pairs / timeframes
    python scripts/download_binance_data.py --symbol ETH/USDC --timeframe 1h
    python scripts/download_binance_data.py --symbol BTC/USDT --start 2025-01-01 --end 2025-12-31
    python scripts/download_binance_data.py --timeframe 1d  # daily directly, no resample
    python scripts/download_binance_data.py --timeframe 1s   # true second-level (slow!)

Timeframe notes:
    - 1m (default): ~220 API requests for 5 months, download in ~2 min
    - 1s: ~13,000 API requests for 5 months, download in ~15 min (use sparingly)
    - 1d: ~5 API requests, near-instant (but lose intraday info)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
BINANCE_RATE_LIMIT_MS = 200          # min ms between kline requests
BINANCE_KLINE_LIMIT = 1000           # max candles per request
CHUNK_CACHE_DIR = Path("git_ignore_folder/.binance_klines_cache")
OUTPUT_DIR = Path("git_ignore_folder/factor_implementation_source_data")
OUTPUT_FILE = "daily_pv.h5"
HDF_KEY = "data"

# Column mapping: ccxt OHLCV → RD-Agent daily_pv columns
_COLUMNS = ["datetime", "$open", "$high", "$low", "$close", "$volume"]
RDAGENT_COLS = ["$open", "$close", "$high", "$low", "$volume", "$factor"]

log = logging.getLogger("download_binance")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _symbol_to_instrument(symbol: str) -> str:
    """Convert CCXT symbol format to RD-Agent instrument name."""
    return symbol.replace("/", "")


def _resample_to_daily(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """
    Resample raw kline DataFrame to daily OHLCV with MultiIndex.

    Parameters
    ----------
    df : pd.DataFrame
        Raw klines with DatetimeIndex and columns [open, high, low, close, volume].
    instrument : str
        Instrument name for the MultiIndex level.

    Returns
    -------
    pd.DataFrame
        Daily resampled data with MultiIndex (datetime, instrument) and
        columns [$open, $close, $high, $low, $volume, $factor].
    """
    daily = df.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    daily.index = pd.to_datetime(daily.index.date)
    daily.index.name = "datetime"
    daily["instrument"] = instrument
    daily = daily.set_index("instrument", append=True).swaplevel(0, 1).sort_index()

    daily = daily.rename(columns={
        "open": "$open",
        "high": "$high",
        "low": "$low",
        "close": "$close",
        "volume": "$volume",
    })
    daily["$factor"] = 1.0
    return daily[RDAGENT_COLS]


def _save_hdf(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to HDF5 with the key RD-Agent expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(path, key=HDF_KEY, mode="w")
    log.info("Saved %s rows to %s", len(df), path)


# ── CCXT download ────────────────────────────────────────────────────────────
def _fetch_klines(exchange, symbol: str, timeframe: str, since_ms: int) -> list | None:
    """Fetch one chunk of klines with retry on transient errors."""
    for attempt in range(3):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=BINANCE_KLINE_LIMIT)
        except ccxt.RateLimitExceeded as e:
            wait = 10 + attempt * 5
            log.warning("Rate limited, waiting %ds...", wait)
            time.sleep(wait)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            wait = 2 + attempt * 2
            log.warning("Exchange/network error (attempt %d): %s — retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)
    log.error("Failed after 3 retries for since=%s", since_ms)
    return None


def download_klines(
    symbol: str,
    timeframe: str = "1m",
    start: str = "2025-11-01",
    end: str = "2026-03-31",
) -> pd.DataFrame:
    """
    Download full kline history from Binance in paginated chunks.

    Uses chunk-level caching to survive interruptions.
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    instrument = _symbol_to_instrument(symbol)
    CHUNK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    start_ms = exchange.parse8601(start + "T00:00:00Z")
    end_ms = exchange.parse8601(end + "T23:59:59Z")
    tf_ms = exchange.parse_timeframe(timeframe) * 1000

    frames = []
    cursor_ms = start_ms
    total_candles = (end_ms - start_ms) // tf_ms
    estimated_requests = (total_candles // BINANCE_KLINE_LIMIT) + 1
    log.info("Downloading %s %s klines: %s -> %s (~%d candles, ~%d API requests)",
             symbol, timeframe, start, end, total_candles, estimated_requests)

    req_count = 0
    while cursor_ms < end_ms:
        # Check cache for this chunk
        chunk_file = CHUNK_CACHE_DIR / f"{instrument}_{timeframe}_{cursor_ms}.parquet"
        if chunk_file.exists():
            chunk = pd.read_parquet(chunk_file)
            if not chunk.empty:
                frames.append(chunk)
                cursor_ms = int(chunk["datetime"].max().timestamp() * 1000) + tf_ms
                log.debug("Loaded cached chunk: %s rows", len(chunk))
                continue

        ohlcv = _fetch_klines(exchange, symbol, timeframe, cursor_ms)
        if ohlcv is None:
            break
        if len(ohlcv) == 0:
            break

        chunk = pd.DataFrame(ohlcv, columns=_COLUMNS)
        chunk["datetime"] = pd.to_datetime(chunk["datetime"], unit="ms", utc=True)
        chunk = chunk.set_index("datetime")

        # Cache chunk
        chunk.reset_index().to_parquet(chunk_file, index=False)
        frames.append(chunk)

        cursor_ms = int(ohlcv[-1][0]) + tf_ms
        req_count += 1

        if req_count % 50 == 0:
            pct = min(100, (cursor_ms - start_ms) / (end_ms - start_ms) * 100)
            log.info("Progress: %.0f%% (%d requests, %d candles)", pct, req_count, sum(len(f) for f in frames))

        time.sleep(BINANCE_RATE_LIMIT_MS / 1000)

    if not frames:
        log.error("No data downloaded. Check symbol/timeframe/date range.")
        sys.exit(1)

    df = pd.concat(frames).sort_index()
    # Drop overlaps from pagination
    df = df[~df.index.duplicated(keep="last")]
    log.info("Downloaded %s candles (%s → %s)", len(df), df.index.min(), df.index.max())
    return df


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download Binance klines → RD-Agent daily_pv.h5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", default="ETH/USDT", help="Trading pair (default: ETH/USDT)")
    parser.add_argument("--timeframe", default="1m",
                        help="Kline interval: 1s, 1m, 5m, 15m, 1h, 4h, 1d (default: 1m)")
    parser.add_argument("--start", default="2025-11-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-03-31", help="End date YYYY-MM-DD")
    parser.add_argument("--output", help="Output .h5 path (default: auto)")
    parser.add_argument("--keep-raw", action="store_true",
                        help="Also save raw klines as parquet")
    parser.add_argument("--no-resample", action="store_true",
                        help="Skip daily resampling (output raw klines as h5)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Validate timeframe
    exchange = ccxt.binance()
    if args.timeframe not in exchange.timeframes:
        log.error("Unsupported timeframe '%s'. Valid: %s", args.timeframe, sorted(exchange.timeframes.keys()))
        sys.exit(1)

    # Download
    t0 = time.time()
    raw = download_klines(args.symbol, args.timeframe, args.start, args.end)
    elapsed = time.time() - t0
    log.info("Download completed in %.0fs", elapsed)

    instrument = _symbol_to_instrument(args.symbol)
    out_path = Path(args.output) if args.output else OUTPUT_DIR / OUTPUT_FILE

    if args.no_resample or args.timeframe == "1d":
        # Convert raw to RD-Agent MultiIndex format
        raw = raw.rename(columns={"open": "$open", "high": "$high", "low": "$low", "close": "$close", "volume": "$volume"})
        raw["$factor"] = 1.0
        raw.index.name = "datetime"
        raw["instrument"] = instrument
        raw = raw.reset_index().set_index(["datetime", "instrument"]).sort_index()
        out = raw[RDAGENT_COLS]
    else:
        out = _resample_to_daily(raw, instrument)

    _save_hdf(out, out_path)

    # Quick sanity check
    log.info("  Columns: %s", list(out.columns))
    log.info("  Index:   %s", list(out.index.names))
    log.info("  Range:   %s → %s", out.index.get_level_values("datetime").min(), out.index.get_level_values("datetime").max())
    log.info("Day count: %d", len(out))

    if args.keep_raw:
        raw_path = Path(f"git_ignore_folder/{instrument}_{args.timeframe}_raw.parquet")
        raw.to_parquet(raw_path)
        log.info("Raw data saved to %s", raw_path)


if __name__ == "__main__":
    main()
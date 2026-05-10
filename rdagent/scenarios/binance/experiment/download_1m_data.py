#!/usr/bin/env python3
"""
Download 1-minute klines for benchmark symbols from Binance data.binance.vision
and convert to Qlib binary format at 1min frequency.

This is needed because Qlib's backtest engine (PortAnaRecord) requires 1min
calendar data internally. Only a few benchmark symbols are needed.

Usage:
    # Download only (generates CSVs):
    python download_1m_data.py --start 2025-01-01 --end 2026-01-31

    # Download + convert to Qlib binary:
    python download_1m_data.py --start 2025-01-01 --end 2026-01-31 \
        --dump-bin /path/to/qlib/scripts/dump_bin.py
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://data.binance.vision"
OUTPUT_DIR = Path(__file__).resolve().parent / "factor_data_template"

log = logging.getLogger("download_1m_data")

# Only benchmark symbols needed for 1min data (Qlib backtest uses these for benchmark)
BENCHMARK_SYMBOLS = ["BTCUSDT"]


def _download_zip(url: str) -> pd.DataFrame | None:
    try:
        resp = urllib.request.urlopen(url)
        data = resp.read()
    except urllib.error.HTTPError:
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
                    df = pd.read_csv(f)
        except zipfile.BadZipFile:
            log.warning("Bad zip: %s", url)
            return None
    return df


def _download_klines_1m(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame | None:
    """Download 1m klines for one symbol. Returns None if any month is missing."""
    chunks = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        filename = f"{symbol}-1m-{current.strftime('%Y-%m')}.zip"
        url = f"{BASE_URL}/data/futures/um/monthly/klines/{symbol}/1m/{filename}"
        df = _download_zip(url)
        if df is None:
            log.warning("  %s %s: not available, skipping", symbol, current.strftime("%Y-%m"))
            current += pd.DateOffset(months=1)
            continue
        chunks.append(df)
        log.info("  %s %s: %d rows", symbol, current.strftime("%Y-%m"), len(df))
        current += pd.DateOffset(months=1)

    if not chunks:
        return None

    full = pd.concat(chunks, ignore_index=True)
    full.columns = [f"col_{i}" for i in range(len(full.columns))]
    full["datetime"] = pd.to_datetime(full["col_0"], unit="ms", utc=True).dt.tz_localize(None)
    full = full.rename(columns={
        "col_1": "open", "col_2": "high", "col_3": "low",
        "col_4": "close", "col_5": "volume",
    })
    full = full.set_index("datetime").sort_index()
    full = full[~full.index.duplicated(keep="first")]
    full = full[["open", "high", "low", "close", "volume"]].astype(np.float64)
    return full.loc[start_dt:end_dt]


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance 1m klines for benchmark symbols -> Qlib 1min binary data"
    )
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-31")
    parser.add_argument("--symbols", nargs="+", default=BENCHMARK_SYMBOLS,
                        help="Symbols to download (default: BTCUSDT)")
    parser.add_argument("--qlib-dir", default=None,
                        help="Output directory for Qlib binary data (default: ~/.qlib/qlib_data/crypto_data)")
    parser.add_argument("--dump-bin", default=None,
                        help="Path to qlib's scripts/dump_bin.py; if omitted, only CSVs are generated")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)
    qlib_dir = Path(args.qlib_dir).expanduser() if args.qlib_dir else Path("~/.qlib/qlib_data/crypto_data").expanduser()

    log.info("Downloading %d symbols at 1m frequency: %s", len(args.symbols), ", ".join(args.symbols))

    csv_dir = OUTPUT_DIR / "_csv_1m_temp"
    csv_dir.mkdir(parents=True, exist_ok=True)

    for symbol in args.symbols:
        log.info("Downloading %s 1m klines...", symbol)
        df = _download_klines_1m(symbol, start_dt, end_dt)
        if df is None or df.empty:
            log.warning("No 1m data for %s, skipping.", symbol)
            continue
        log.info("Finished %s: %d rows", symbol, len(df))

        # Write CSV for dump_bin.py
        csv_df = df.reset_index()
        csv_df["symbol"] = symbol
        csv_df[["symbol", "datetime", "open", "high", "low", "close", "volume"]].rename(
            columns={"datetime": "date"}
        ).to_csv(csv_dir / f"{symbol}.csv", index=False)

    if not any(csv_dir.glob("*.csv")):
        log.error("No data downloaded.")
        raise SystemExit(1)

    log.info("CSV files written to %s", csv_dir)

    if args.dump_bin:
        dump_bin = Path(args.dump_bin)
        if not dump_bin.exists():
            raise FileNotFoundError(f"dump_bin.py not found: {dump_bin}")
        cmd = [
            sys.executable, str(dump_bin), "dump_all",
            "--data_path", str(csv_dir),
            "--qlib_dir", str(qlib_dir),
            "--include_fields", "open,high,low,close,volume",
            "--symbol_field_name", "symbol",
            "--date_field_name", "date",
            "--freq", "1min",
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        shutil.rmtree(csv_dir, ignore_errors=True)
        print(f"Qlib 1min binary data generated at {qlib_dir}")
    else:
        print(
            "\nCSV files are ready. To generate Qlib 1min binary data, run:\n\n"
            f"  python dump_bin.py dump_all \\\n"
            f"    --data_path {csv_dir} \\\n"
            f"    --qlib_dir {qlib_dir} \\\n"
            f"    --include_fields open,high,low,close,volume \\\n"
            f"    --symbol_field_name symbol \\\n"
            f"    --date_field_name date \\\n"
            f"    --freq 1min\n\n"
            "NOTE: This adds 1min data to the same qlib_dir. The 60min data should already exist there."
        )


if __name__ == "__main__":
    main()

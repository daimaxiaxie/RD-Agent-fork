#!/usr/bin/env python3
"""
Download multi-coin 1h klines from Binance data.binance.vision (perpetual futures only)
and save as HDF5 for the binance_factor scenario.

Output files:
  - hourly_pv_all.h5       full dataset (key="data")
  - hourly_pv_debug.h5     subset: first 5 coins, 6 months around midpoint

Index:  MultiIndex (datetime, instrument)
Columns: $open, $high, $low, $close, $volume

Usage:
    python download_data.py
    python download_data.py --start 2021-01-01 --end 2024-12-31
    # After download, generate Qlib binary data separately:
    python generate_qlib_data.py
"""

import argparse
import logging
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://data.binance.vision"
OUTPUT_DIR = Path(__file__).resolve().parent / "factor_data_template"

log = logging.getLogger("download_binance_data")

SYMBOLS = [
    # 压舱石
    "BTCUSDT", "ETHUSDT", "BNBUSDT",
    # 强动能
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "DOTUSDT", "LINKUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT",
    "TRXUSDT", "TONUSDT", "FTMUSDT", "AAVEUSDT",
    "CRVUSDT", "SNXUSDT", "GRTUSDT", "FILUSDT", "STXUSDT",
    "IMXUSDT", "ETCUSDT", "BCHUSDT", "SANDUSDT", "ALGOUSDT",
    "ICPUSDT", "FLOWUSDT", "THETAUSDT", "AXSUSDT", 
    "GALAUSDT", "1000PEPEUSDT", "PENDLEUSDT", "LDOUSDT", "ENSUSDT", 
    "PYTHUSDT", "IOTAUSDT", "AXLUSDT", "EGLDUSDT", "CKBUSDT",
    "AEROUSDT", "KAIAUSDT", "DRIFTUSDT", 
    # 增强器
    "OPUSDT", "ARBUSDT", "APTUSDT", "NEARUSDT", "INJUSDT",
    "TIAUSDT", "SUIUSDT", "FETUSDT", "WLDUSDT",
    "CFXUSDT", "MASKUSDT", "DYDXUSDT", "1000FLOKIUSDT", "ONDOUSDT",
    "BOMEUSDT", "MEMEUSDT", "MKRUSDT", "COMPUSDT", "ENAUSDT",
    "GOATUSDT", "NEIROUSDT", "MYROUSDT", "JUPUSDT", "WIFUSDT",
]


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


def _download_klines_1h(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    """Download 1h klines for one symbol. Returns None if any month is missing."""
    chunks = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        filename = f"{symbol}-1h-{current.strftime('%Y-%m')}.zip"
        url = f"{BASE_URL}/data/futures/um/monthly/klines/{symbol}/1h/{filename}"
        df = _download_zip(url)
        if df is None:
            log.warning("  %s %s: not available, skipping symbol", symbol, current.strftime("%Y-%m"))
            return None
        chunks.append(df)
        log.info("  %s %s: %d rows", symbol, current.strftime("%Y-%m"), len(df))
        current += pd.DateOffset(months=1)

    full = pd.concat(chunks, ignore_index=True)
    full.columns = [f"col_{i}" for i in range(len(full.columns))]
    full["datetime"] = pd.to_datetime(full["col_0"], unit="ms", utc=True).dt.tz_localize(None)
    full = full.rename(columns={
        "col_1": "$open", "col_2": "$high", "col_3": "$low",
        "col_4": "$close", "col_5": "$volume",
    })
    full = full.set_index("datetime").sort_index()
    full = full[~full.index.duplicated(keep="first")]
    full = full[["$open", "$high", "$low", "$close", "$volume"]].astype(np.float64)
    return full.loc[start_dt:end_dt]


def main():
    parser = argparse.ArgumentParser(description="Download Binance perpetual futures 1h klines -> hourly_pv.h5")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output", help="Output path (default: factor_data_template/hourly_pv_all.h5)")
    parser.add_argument("--debug-output", help="Debug path (default: factor_data_template/hourly_pv_debug.h5)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)
    log.info("Downloading %d symbols with 4 threads: %s", len(SYMBOLS), ", ".join(SYMBOLS))

    all_frames = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_download_klines_1h, s, start_dt, end_dt): s for s in SYMBOLS}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
            except Exception as e:
                log.warning("Error downloading %s: %s", symbol, e)
                continue
            if df is None or df.empty:
                log.warning("No data for %s, skipping.", symbol)
                continue
            log.info("Finished %s", symbol)
            df = df.copy()
            df["instrument"] = symbol
            df = df.set_index([df.index, "instrument"])
            df.index.names = ["datetime", "instrument"]
            all_frames.append(df)

    if not all_frames:
        log.error("No data downloaded.")
        raise SystemExit(1)

    combined = pd.concat(all_frames).sort_index().dropna(how="all")

    log.info("Combined: %d rows, %d instruments, %s -> %s",
             len(combined),
             combined.index.get_level_values("instrument").nunique(),
             combined.index.get_level_values("datetime").min(),
             combined.index.get_level_values("datetime").max())

    # Save full (HDF5/PyTables does not support category in MultiIndex)
    out_path = Path(args.output) if args.output else OUTPUT_DIR / "hourly_pv_all.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_hdf(out_path, key="data", mode="w")
    log.info("Saved to %s", out_path)

    # Save debug (5 coins, 6 months around midpoint)
    debug_path = Path(args.debug_output) if args.debug_output else OUTPUT_DIR / "hourly_pv_debug.h5"
    debug_coins = combined.index.get_level_values("instrument").unique()[:5]
    dt_min = combined.index.get_level_values("datetime").min()
    dt_max = combined.index.get_level_values("datetime").max()
    mid = dt_min + (dt_max - dt_min) / 2
    debug_df = combined.loc[
        combined.index.get_level_values("instrument").isin(debug_coins) &
        (combined.index.get_level_values("datetime") >= mid - pd.Timedelta(days=90)) &
        (combined.index.get_level_values("datetime") <= mid + pd.Timedelta(days=90))
    ]
    debug_df.to_hdf(debug_path, key="data", mode="w")
    log.info("Saved debug (%d rows, %d coins) to %s", len(debug_df), len(debug_coins), debug_path)

    log.info("Download complete. To generate Qlib binary data, run:\n"
             "  python rdagent/scenarios/binance/experiment/generate_qlib_data.py")


if __name__ == "__main__":
    main()
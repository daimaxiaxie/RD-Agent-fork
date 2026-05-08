#!/usr/bin/env python3
"""
Convert Binance hourly HDF5 data to Qlib binary format.

This script is designed to run inside the Qlib Docker/conda environment
(where qlib is installed). It reads hourly_pv.h5 from factor_data_template/,
converts it to per-instrument CSVs, and calls DumpDataAll to produce
Qlib binary data at ~/.qlib/qlib_data/crypto_data.

Usage (inside qlib env):
    python generate_qlib_data.py
"""

import shutil
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "factor_data_template"
H5_PATH = DATA_DIR / "hourly_pv_all.h5"
QLIB_DATA_PATH = Path("~/.qlib/qlib_data/crypto_data").expanduser()


def main():
    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {H5_PATH}\n"
            "Please run download_data.py first:\n"
            "  python rdagent/scenarios/binance/experiment/download_data.py --start 2024-01-01 --end 2025-06-30"
        )

    df = pd.read_hdf(H5_PATH, key="data")

    # Rename columns: $open -> open, etc.
    df.columns = [c.lstrip("$") for c in df.columns]

    # Flatten MultiIndex to columns for CSV export
    df = df.reset_index()
    df = df.rename(columns={"datetime": "date", "instrument": "symbol"})

    # Write per-symbol CSV files to a temp directory
    csv_dir = DATA_DIR / "_csv_temp"
    csv_dir.mkdir(exist_ok=True)
    for symbol, group in df.groupby("symbol"):
        group[["symbol", "date", "open", "high", "low", "close", "volume"]].to_csv(
            csv_dir / f"{symbol}.csv", index=False
        )

    # Convert to Qlib binary format
    from qlib.dump_bin import DumpDataAll

    dumper = DumpDataAll(
        csv_path=str(csv_dir),
        provider_uri=str(QLIB_DATA_PATH),
        include_fields="open,high,low,close,volume",
        symbol_field_name="symbol",
        date_field_name="date",
    )
    dumper.dump()

    # Cleanup temp CSVs
    shutil.rmtree(csv_dir)
    print(f"Qlib crypto data dumped to {QLIB_DATA_PATH}")


if __name__ == "__main__":
    main()

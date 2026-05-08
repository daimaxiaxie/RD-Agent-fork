#!/usr/bin/env python3
"""
Convert downloaded Binance hourly HDF5 data to Qlib binary format.

Must be run inside a qlib-enabled environment (Docker/conda).
Requires the data file already downloaded via download_data.py.

Usage:
    python generate_qlib_data.py
    python generate_qlib_data.py --h5 path/to/hourly_pv_all.h5 --qlib-dir ~/.qlib/qlib_data/crypto_data
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def generate_qlib_data(h5_path: Path, qlib_dir: Path) -> None:
    from qlib.dump_bin import DumpDataAll

    df = pd.read_hdf(h5_path, key="data")
    df.columns = [c.lstrip("$") for c in df.columns]
    df = df.reset_index().rename(columns={"datetime": "date", "instrument": "symbol"})

    csv_dir = h5_path.parent / "_csv_temp"
    csv_dir.mkdir(exist_ok=True)
    for symbol, grp in df.groupby("symbol"):
        grp[["symbol", "date", "open", "high", "low", "close", "volume"]].to_csv(
            csv_dir / f"{symbol}.csv", index=False
        )

    DumpDataAll(
        csv_path=str(csv_dir),
        provider_uri=str(qlib_dir),
        include_fields="open,high,low,close,volume",
        symbol_field_name="symbol",
        date_field_name="date",
    ).dump()
    shutil.rmtree(csv_dir, ignore_errors=True)
    print(f"Qlib binary data generated at {qlib_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Binance HDF5 data to Qlib binary format")
    parser.add_argument(
        "--h5",
        default=None,
        help="Path to hourly_pv_all.h5 (default: factor_data_template/hourly_pv_all.h5)",
    )
    parser.add_argument(
        "--qlib-dir",
        default=None,
        help="Output directory for Qlib binary data (default: ~/.qlib/qlib_data/crypto_data)",
    )
    args = parser.parse_args()

    default_h5 = Path(__file__).resolve().parent / "factor_data_template" / "hourly_pv_all.h5"
    h5_path = Path(args.h5) if args.h5 else default_h5
    qlib_dir = Path(args.qlib_dir).expanduser() if args.qlib_dir else Path("~/.qlib/qlib_data/crypto_data").expanduser()

    if not h5_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {h5_path}\n"
            "Please run download_data.py first:\n"
            "  python rdagent/scenarios/binance/experiment/download_data.py"
        )

    generate_qlib_data(h5_path, qlib_dir)


if __name__ == "__main__":
    main()

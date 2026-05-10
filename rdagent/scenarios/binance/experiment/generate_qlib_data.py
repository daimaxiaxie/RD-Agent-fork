#!/usr/bin/env python3
"""
Convert downloaded Binance hourly HDF5 data to Qlib binary format.

Step 1: Convert HDF5 -> per-symbol CSVs (pure pandas, no qlib needed).
Step 2: Call qlib's scripts/dump_bin.py to convert CSVs -> Qlib binary data.
        Requires qlib repo scripts on PYTHONPATH in a qlib-enabled environment.

Usage:
    # Step 1 only (generates CSVs, prints the command for step 2):
    python generate_qlib_data.py

    # Step 1 + 2 (if qlib scripts are available):
    python generate_qlib_data.py --dump-bin /path/to/qlib/scripts/dump_bin.py
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

QLIB_DUMP_BIN_URL = "https://github.com/microsoft/qlib/blob/main/scripts/dump_bin.py"


def h5_to_csv(h5_path: Path, csv_dir: Path) -> None:
    df = pd.read_hdf(h5_path, key="data")
    df.columns = [c.lstrip("$") for c in df.columns]
    df = df.reset_index().rename(columns={"datetime": "date", "instrument": "symbol"})

    csv_dir.mkdir(parents=True, exist_ok=True)
    for symbol, grp in df.groupby("symbol"):
        grp[["symbol", "date", "open", "high", "low", "close", "volume"]].to_csv(
            csv_dir / f"{symbol}.csv", index=False
        )
    print(f"CSV files written to {csv_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Binance HDF5 data to Qlib binary format")
    parser.add_argument("--h5", default=None, help="Path to hourly_pv_all.h5 (default: factor_data_template/hourly_pv_all.h5)")
    parser.add_argument("--qlib-dir", default=None, help="Output directory for Qlib binary data (default: ~/.qlib/qlib_data/crypto_data)")
    parser.add_argument("--dump-bin", default=None, help="Path to qlib's scripts/dump_bin.py; if omitted, only CSVs are generated")
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

    csv_dir = h5_path.parent / "_csv_temp"
    h5_to_csv(h5_path, csv_dir)

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
            "--freq", "60min",
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        shutil.rmtree(csv_dir, ignore_errors=True)
        print(f"Qlib binary data generated at {qlib_dir}")
    else:
        print(
            "\nCSV files are ready. To generate Qlib binary data, run in your qlib environment:\n\n"
            f"  python dump_bin.py dump_all \\\n"
            f"    --data_path {csv_dir} \\\n"
            f"    --qlib_dir {qlib_dir} \\\n"
            f"    --include_fields open,high,low,close,volume \\\n"
            f"    --symbol_field_name symbol \\\n"
            f"    --date_field_name date \\\n"
            f"    --freq 60min\n\n"
            f"Get dump_bin.py from: {QLIB_DUMP_BIN_URL}"
        )


if __name__ == "__main__":
    main()
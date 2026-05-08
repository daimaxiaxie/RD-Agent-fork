"""
Prepare Binance factor data for factor execution.

Step 1: Copies downloaded hourly PV data into the data folders
        used by the factor execution pipeline.
Step 2: Generates Qlib binary data (requires qlib environment).

Usage:
    python generate.py
"""

import shutil
from pathlib import Path

from rdagent.app.binance_rd_loop.conf import BINANCE_FACTOR_PROP_SETTING
from rdagent.scenarios.binance.experiment.download_data import generate_qlib_data

TEMPLATE_DIR = Path(__file__).resolve().parent


def main():
    h5_all = TEMPLATE_DIR / "hourly_pv_all.h5"
    h5_debug = TEMPLATE_DIR / "hourly_pv_debug.h5"
    readme = TEMPLATE_DIR / "README.md"

    if not h5_all.exists():
        raise FileNotFoundError(
            f"Data file not found: {h5_all}\n"
            "Please run download_data.py first:\n"
            "  python rdagent/scenarios/binance/experiment/download_data.py --start 2024-01-01 --end 2025-06-30"
        )

    # Copy to data_folder
    data_folder = Path(BINANCE_FACTOR_PROP_SETTING.data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(h5_all, data_folder / "hourly_pv.h5")
    shutil.copy2(readme, data_folder / "README.md")
    print(f"Copied data to {data_folder}")

    # Copy debug data
    data_folder_debug = Path(BINANCE_FACTOR_PROP_SETTING.data_folder_debug)
    data_folder_debug.mkdir(parents=True, exist_ok=True)
    if h5_debug.exists():
        shutil.copy2(h5_debug, data_folder_debug / "hourly_pv.h5")
    else:
        shutil.copy2(h5_all, data_folder_debug / "hourly_pv.h5")
    shutil.copy2(readme, data_folder_debug / "README.md")
    print(f"Copied debug data to {data_folder_debug}")

    # Generate Qlib binary data (requires qlib, skips gracefully if not available)
    qlib_dir = Path(BINANCE_FACTOR_PROP_SETTING.qlib_provider_uri).expanduser()
    if not qlib_dir.exists():
        generate_qlib_data(h5_all, qlib_dir)
    print("Done.")


if __name__ == "__main__":
    main()

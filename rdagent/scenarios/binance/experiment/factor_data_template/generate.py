"""
Prepare Binance factor data for factor execution.

Step 1: Symlinks downloaded PV data into the data folders
        used by the factor execution pipeline.
Step 2: Generates Qlib binary data (requires qlib environment).

Usage:
    python generate.py
"""

import os
from pathlib import Path

from rdagent.app.binance_rd_loop.conf import BINANCE_FACTOR_PROP_SETTING
from rdagent.scenarios.binance.experiment.generate_qlib_data import generate_qlib_data

TEMPLATE_DIR = Path(__file__).resolve().parent


def _symlink(src: Path, dst: Path) -> None:
    dst = dst.resolve()
    if dst.exists() or dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main():
    h5_all = TEMPLATE_DIR / "pv_all.h5"
    h5_debug = TEMPLATE_DIR / "pv_debug.h5"
    readme = TEMPLATE_DIR / "README.md"

    # Fallback to legacy filenames for backwards compatibility
    if not h5_all.exists():
        h5_all = TEMPLATE_DIR / "hourly_pv_all.h5"
    if not h5_debug.exists():
        h5_debug = TEMPLATE_DIR / "hourly_pv_debug.h5"

    if not h5_all.exists():
        raise FileNotFoundError(
            f"Data file not found: {h5_all}\n"
            "Please run download_data.py first:\n"
            "  python rdagent/scenarios/binance/experiment/download_data.py --interval 1h"
        )

    # Symlink to data_folder
    data_folder = Path(BINANCE_FACTOR_PROP_SETTING.data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)
    _symlink(h5_all, data_folder / "pv.h5")
    _symlink(readme, data_folder / "README.md")
    print(f"Linked data to {data_folder}")

    # Symlink debug data
    data_folder_debug = Path(BINANCE_FACTOR_PROP_SETTING.data_folder_debug)
    data_folder_debug.mkdir(parents=True, exist_ok=True)
    _symlink(h5_debug if h5_debug.exists() else h5_all, data_folder_debug / "pv.h5")
    _symlink(readme, data_folder_debug / "README.md")
    print(f"Linked debug data to {data_folder_debug}")

    # Generate Qlib binary data (requires qlib, skips gracefully if not available)
    qlib_dir = Path(BINANCE_FACTOR_PROP_SETTING.qlib_provider_uri).expanduser()
    if not qlib_dir.exists():
        generate_qlib_data(h5_all, qlib_dir, freq=BINANCE_FACTOR_PROP_SETTING.freq)
    print("Done.")


if __name__ == "__main__":
    main()

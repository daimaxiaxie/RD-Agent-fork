import os
import random
import re
import shutil
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.app.binance_rd_loop.conf import BINANCE_FACTOR_PROP_SETTING
from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.utils.env import QTDockerEnv, QlibCondaEnv, QlibCondaConf


TEMPLATE_DIR = Path(__file__).parent / "factor_data_template"


def generate_data_folder():
    """
    Prepare the data folders for Binance factor execution.

    1. Check if factor_data_template/hourly_pv_all.h5 exists (downloaded by download_data.py).
    2. Copy data to the data_folder and data_folder_debug directories.
    3. Run generate_qlib_data.py inside Docker/conda to create Qlib binary data.
    """
    data_folder = Path(BINANCE_FACTOR_PROP_SETTING.data_folder)
    data_folder_debug = Path(BINANCE_FACTOR_PROP_SETTING.data_folder_debug)

    h5_all = TEMPLATE_DIR / "hourly_pv_all.h5"
    h5_debug = TEMPLATE_DIR / "hourly_pv_debug.h5"
    readme = TEMPLATE_DIR / "README.md"

    # Check source data exists
    if not h5_all.exists():
        raise FileNotFoundError(
            f"Binance hourly data not found at {h5_all}\n"
            "Please run the download script first:\n"
            "  python rdagent/scenarios/binance/experiment/download_data.py --start 2024-01-01 --end 2025-06-30"
        )

    # Copy data to data_folder (used by factor execution)
    data_folder.mkdir(parents=True, exist_ok=True)
    dst = data_folder / "hourly_pv.h5"
    if not dst.exists() or os.path.getmtime(h5_all) > os.path.getmtime(dst):
        shutil.copy2(h5_all, dst)
    readme_dst = data_folder / "README.md"
    if not readme_dst.exists():
        shutil.copy2(readme, readme_dst)

    # Copy debug data
    data_folder_debug.mkdir(parents=True, exist_ok=True)
    dst_debug = data_folder_debug / "hourly_pv.h5"
    if h5_debug.exists():
        if not dst_debug.exists() or os.path.getmtime(h5_debug) > os.path.getmtime(dst_debug):
            shutil.copy2(h5_debug, dst_debug)
    else:
        # If no debug file, copy the full one
        if not dst_debug.exists():
            shutil.copy2(h5_all, dst_debug)
    readme_dst_debug = data_folder_debug / "README.md"
    if not readme_dst_debug.exists():
        shutil.copy2(readme, readme_dst_debug)

    # Generate Qlib binary data if not already present
    qlib_provider = Path(BINANCE_FACTOR_PROP_SETTING.qlib_provider_uri).expanduser()
    if not qlib_provider.exists():
        if MODEL_COSTEER_SETTINGS.env_type == "docker":
            qtde = QTDockerEnv()
        elif MODEL_COSTEER_SETTINGS.env_type == "conda":
            qtde = QlibCondaEnv(conf=QlibCondaConf())
        else:
            raise ValueError(f"Unknown env_type: {MODEL_COSTEER_SETTINGS.env_type}")
        qtde.prepare()

        qtde.check_output(
            local_path=str(TEMPLATE_DIR.parent),
            entry="python generate_qlib_data.py",
        )

        assert qlib_provider.exists(), (
            f"Qlib crypto data not generated at {qlib_provider}. "
            "Check generate_qlib_data.py output for errors."
        )


def get_file_desc(p: Path, variable_list=[]) -> str:
    p = Path(p)

    JJ_TPL = Environment(undefined=StrictUndefined).from_string("""
# {{file_name}}

## File Type
{{type_desc}}

## Content Overview
{{content}}
""")

    if p.name.endswith(".h5"):
        df = pd.read_hdf(p)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)

        df_info = "### Data Structure\n"
        df_info += (
            f"- Index: MultiIndex with levels {df.index.names}\n"
            if isinstance(df.index, pd.MultiIndex)
            else f"- Index: {df.index.name}\n"
        )

        df_info += "\n### Columns\n"
        columns = df.dtypes.to_dict()
        grouped_columns = {}

        for col in columns:
            if col.startswith("$"):
                prefix = col.split("_")[0] if "_" in col else col
                grouped_columns.setdefault(prefix, []).append(col)
            else:
                grouped_columns.setdefault("other", []).append(col)

        if variable_list:
            df_info += "#### Relevant Columns:\n"
            relevant_line = ", ".join(f"{col}: {columns[col]}" for col in variable_list if col in columns)
            df_info += relevant_line + "\n"
        else:
            df_info += "#### All Columns:\n"
            grouped_items = list(grouped_columns.items())
            random.shuffle(grouped_items)
            for prefix, cols in grouped_items:
                header = "Other Columns" if prefix == "other" else f"{prefix} Related Columns"
                df_info += f"\n#### {header}:\n"
                random.shuffle(cols)
                line = ", ".join(f"{col}: {columns[col]}" for col in cols)
                df_info += line + "\n"

        return JJ_TPL.render(
            file_name=p.name,
            type_desc="HDF5 Data File",
            content=df_info,
        )

    elif p.name.endswith(".md"):
        with open(p) as f:
            content = f.read()
            return JJ_TPL.render(
                file_name=p.name,
                type_desc="Markdown Documentation",
                content=content,
            )

    else:
        raise NotImplementedError(
            f"file type {p.name} is not supported. Please implement its description function.",
        )


def get_data_folder_intro(fname_reg: str = ".*", flags=0, variable_mapping=None) -> str:
    """
    Get the info of the data folder for prompting messages.
    Auto-generates the data folder if it doesn't exist.
    """
    data_folder = Path(BINANCE_FACTOR_PROP_SETTING.data_folder)
    data_folder_debug = Path(BINANCE_FACTOR_PROP_SETTING.data_folder_debug)

    if not data_folder.exists() or not data_folder_debug.exists():
        generate_data_folder()

    content_l = []
    for p in data_folder_debug.iterdir():
        if re.match(fname_reg, p.name, flags) is not None:
            if variable_mapping:
                content_l.append(get_file_desc(p, variable_mapping.get(p.stem, [])))
            else:
                content_l.append(get_file_desc(p))
    return "\n----------------- file splitter -------------\n".join(content_l)

import random
import re
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.app.binance_rd_loop.conf import BINANCE_FACTOR_PROP_SETTING


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
    Raises FileNotFoundError if data has not been prepared.
    """
    data_folder = Path(BINANCE_FACTOR_PROP_SETTING.data_folder)
    data_folder_debug = Path(BINANCE_FACTOR_PROP_SETTING.data_folder_debug)
    qlib_provider = Path(BINANCE_FACTOR_PROP_SETTING.qlib_provider_uri).expanduser()

    missing = []
    if not data_folder.exists():
        missing.append(f"  - {data_folder} (factor execution data)")
    if not data_folder_debug.exists():
        missing.append(f"  - {data_folder_debug} (debug data)")
    if not qlib_provider.exists():
        missing.append(f"  - {qlib_provider} (Qlib binary data)")

    if missing:
        raise FileNotFoundError(
            "Binance factor data not found. Missing:\n"
            + "\n".join(missing)
            + "\n\nPlease prepare the data first:\n"
            "  # Step 1: Download kline data\n"
            "  python rdagent/scenarios/binance/experiment/download_data.py --start 2024-01-01 --end 2025-06-30\n\n"
            "  # Step 2: Copy data to working directories\n"
            "  python rdagent/scenarios/binance/experiment/factor_data_template/generate.py\n\n"
            "  # Step 3: Generate Qlib binary data (requires qlib environment)\n"
            "  python rdagent/scenarios/binance/experiment/download_data.py --qlib"
        )

    content_l = []
    for p in data_folder_debug.iterdir():
        if re.match(fname_reg, p.name, flags) is not None:
            if variable_mapping:
                content_l.append(get_file_desc(p, variable_mapping.get(p.stem, [])))
            else:
                content_l.append(get_file_desc(p))
    return "\n----------------- file splitter -------------\n".join(content_l)

#!/usr/bin/env python3
"""
Post-run CLI that reads an rdagent log directory and prints a ranked
summary of all factors discovered during a factor evolution run.

Supports both ethusdt_factor and qlib fin_factor scenarios.

Usage:
    python scripts/factor_summary.py <log_dir>
    python scripts/factor_summary.py <log_dir> --sort-by "Rank IC" --top 10
    python scripts/factor_summary.py <log_dir> --sota-only --show-code
    python scripts/factor_summary.py <log_dir> --export factor_report.json
    python scripts/factor_summary.py <log_dir> --detail
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

# ── Helpers ────────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


def _load_trace_from_session(log_dir: Path):
    """Load the latest session pickle and return the trace object."""
    session_folder = log_dir / "__session__"
    if not session_folder.exists():
        return None
    files = sorted(
        session_folder.glob("*/*_*"),
        key=lambda f: (int(f.parent.name), int(f.name.split("_")[0])),
    )
    if not files:
        return None
    with files[-1].open("rb") as f:
        loop_obj = pickle.load(f)
    return getattr(loop_obj, "trace", None)


def _extract_workspace_paths(exp) -> dict:
    """Extract all relevant file paths from an experiment."""
    paths = {}
    ws = getattr(exp, "experiment_workspace", None)
    if ws is not None:
        ws_path = getattr(ws, "workspace_path", None)
        if ws_path is not None:
            paths["workspace_dir"] = str(ws_path)
            # Standard files expected in workspace after running
            for fname in [
                "combined_factors_df.parquet",
                "crypto_dataset.parquet",
                "conf_baseline.yaml",
                "conf_combined_factors.yaml",
                "qlib_res.csv",
                "ret.pkl",
                "pred.pkl",
            ]:
                fpath = ws_path / fname
                if fpath.exists():
                    paths[fname] = str(fpath)
            # Check for Qlib output subdirectories
            for subdir in ["portfolio_analysis", "signals", "signal", "mlflow"]:
                sd = ws_path / subdir
                if sd.exists() and sd.is_dir():
                    paths[f"{subdir}/"] = str(sd)

    # Factor code workspaces (individual factor.py locations)
    sub_ws = getattr(exp, "sub_workspace_list", []) or []
    sub_tasks = getattr(exp, "sub_tasks", []) or []
    factor_workspaces = []
    for task_idx, task in enumerate(sub_tasks):
        if task_idx < len(sub_ws) and sub_ws[task_idx] is not None:
            fws = sub_ws[task_idx]
            fws_path = getattr(fws, "workspace_path", None)
            fname = getattr(task, "factor_name", None) or getattr(task, "name", f"task_{task_idx}")
            fw_info = {"factor_name": fname}
            if fws_path is not None:
                fw_info["workspace_dir"] = str(fws_path)
                factor_py = fws_path / "factor.py"
                if factor_py.exists():
                    fw_info["factor.py"] = str(factor_py)
                result_h5 = fws_path / "result.h5"
                if result_h5.exists():
                    fw_info["result.h5"] = str(result_h5)
            # Also check file_dict for factor.py even if workspace doesn't exist on disk
            if fws.file_dict and "factor.py" in fws.file_dict:
                fw_info["factor_code"] = fws.file_dict["factor.py"]
            factor_workspaces.append(fw_info)
    paths["factor_workspaces"] = factor_workspaces

    # Base features / feature codes from the experiment
    base_features = getattr(exp, "base_features", {})
    base_feature_codes = getattr(exp, "base_feature_codes", {})
    if base_features:
        paths["base_features"] = base_features
    if base_feature_codes:
        paths["base_feature_codes"] = base_feature_codes

    return paths


def _extract_factors(trace) -> tuple[list[dict], dict | None]:
    """Walk trace.hist and extract factor records. Also return SOTA details."""
    factors = []
    sota_detail = None

    for idx, (exp, fb) in enumerate(trace.hist):
        loop_id = trace.idx2loop_id.get(idx, idx)
        parent_idx = trace.dag_parent[idx] if idx < len(trace.dag_parent) else ()
        parent_loop = trace.idx2loop_id.get(parent_idx[0], None) if parent_idx else None

        # Metrics
        result = exp.result
        metrics = {}
        if result is not None and isinstance(result, pd.Series):
            for m in ["IC", "Rank IC", "RMSE"]:
                if m in result.index:
                    v = result[m]
                    metrics[m] = float(v) if pd.notna(v) else None
        elif result is not None and isinstance(result, pd.DataFrame):
            try:
                for m in ["IC", "Rank IC", "RMSE"]:
                    if m in result.index:
                        v = result.loc[m].iloc[0]
                        metrics[m] = float(v) if pd.notna(v) else None
            except Exception:
                pass

        is_sota = bool(fb.decision) if fb is not None else False
        hypothesis_text = ""
        if exp.hypothesis is not None:
            hypothesis_text = getattr(exp.hypothesis, "hypothesis", str(exp.hypothesis)) or ""
        feedback_reason = getattr(fb, "reason", "") if fb is not None else ""
        feedback_observations = getattr(fb, "observations", None) if fb is not None else None

        # Workspace paths
        paths = _extract_workspace_paths(exp)

        # Factor tasks and code
        sub_tasks = getattr(exp, "sub_tasks", []) or []
        sub_ws = getattr(exp, "sub_workspace_list", []) or []

        for task_idx, task in enumerate(sub_tasks):
            factor_name = getattr(task, "factor_name", None) or getattr(task, "name", f"task_{task_idx}")
            factor_desc = getattr(task, "factor_description", None) or getattr(task, "description", "")
            factor_formula = getattr(task, "factor_formulation", "")
            variables = getattr(task, "variables", {})

            factor_code = None
            factor_py_path = None
            if task_idx < len(sub_ws) and sub_ws[task_idx] is not None:
                ws = sub_ws[task_idx]
                factor_code = ws.file_dict.get("factor.py", None) if ws.file_dict else None
                fws_path = getattr(ws, "workspace_path", None)
                if fws_path is not None:
                    fp = fws_path / "factor.py"
                    if fp.exists():
                        factor_py_path = str(fp)

            factors.append({
                "loop_id": loop_id,
                "trace_idx": idx,
                "parent_loop_id": parent_loop,
                "factor_name": factor_name,
                "factor_description": factor_desc,
                "factor_formulation": factor_formula,
                "variables": str(variables) if variables else "",
                "factor_code": factor_code,
                "factor_py_path": factor_py_path,
                "IC": metrics.get("IC"),
                "Rank IC": metrics.get("Rank IC"),
                "RMSE": metrics.get("RMSE"),
                "is_sota": is_sota,
                "hypothesis": hypothesis_text,
                "feedback_reason": feedback_reason,
            })

        # Track the latest SOTA experiment
        if is_sota:
            sota_detail = {
                "loop_id": loop_id,
                "trace_idx": idx,
                "metrics": metrics,
                "hypothesis": hypothesis_text,
                "feedback_reason": feedback_reason,
                "feedback_observations": feedback_observations,
                "paths": paths,
                "factor_names": [
                    getattr(t, "factor_name", None) or getattr(t, "name", f"task_{i}")
                    for i, t in enumerate(sub_tasks)
                ],
            }

    # If no sub_tasks but the experiment has a result, add one entry per loop
    if not factors and len(trace.hist) > 0:
        for idx, (exp, fb) in enumerate(trace.hist):
            loop_id = trace.idx2loop_id.get(idx, idx)
            result = exp.result
            metrics = {}
            if result is not None and isinstance(result, pd.Series):
                for m in ["IC", "Rank IC", "RMSE"]:
                    if m in result.index:
                        v = result[m]
                        metrics[m] = float(v) if pd.notna(v) else None

            is_sota = bool(fb.decision) if fb is not None else False
            hypothesis_text = ""
            if exp.hypothesis is not None:
                hypothesis_text = getattr(exp.hypothesis, "hypothesis", str(exp.hypothesis)) or ""
            feedback_reason = getattr(fb, "reason", "") if fb is not None else ""

            paths = _extract_workspace_paths(exp)

            factors.append({
                "loop_id": loop_id,
                "trace_idx": idx,
                "parent_loop_id": None,
                "factor_name": f"Loop_{loop_id}",
                "factor_description": hypothesis_text[:200] if hypothesis_text else "",
                "factor_formulation": "",
                "variables": "",
                "factor_code": None,
                "factor_py_path": None,
                "IC": metrics.get("IC"),
                "Rank IC": metrics.get("Rank IC"),
                "RMSE": metrics.get("RMSE"),
                "is_sota": is_sota,
                "hypothesis": hypothesis_text,
                "feedback_reason": feedback_reason,
            })

            if is_sota:
                sota_detail = {
                    "loop_id": loop_id,
                    "trace_idx": idx,
                    "metrics": metrics,
                    "hypothesis": hypothesis_text,
                    "feedback_reason": feedback_reason,
                    "feedback_observations": None,
                    "paths": paths,
                    "factor_names": [f"Loop_{loop_id}"],
                }

    return factors, sota_detail


def _load_from_json(log_dir: Path) -> tuple[list[dict] | None, dict | None]:
    """Load factor_summary.json if it exists (written by the loop enhancement)."""
    json_path = log_dir / "factor_summary.json"
    if not json_path.exists():
        return None, None
    with json_path.open("r") as f:
        data = json.load(f)
    factors = data.get("factors", [])
    sota = data.get("sota_detail", None)
    return factors, sota


def _fmt(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.6f}"


def _print_sota_detail(sota: dict, no_color: bool):
    """Print the SOTA experiment detail section."""
    b = "" if no_color else BOLD
    g = "" if no_color else GREEN
    c = "" if no_color else CYAN
    y = "" if no_color else YELLOW
    d = "" if no_color else DIM
    rs = "" if no_color else RESET

    print(f"\n{b}{'=' * 60}")
    print(f"  SOTA Experiment (Loop {sota.get('loop_id', '?')})")
    print(f"{'=' * 60}{rs}\n")

    # Metrics
    metrics = sota.get("metrics", {})
    print(f"  {b}Metrics:{rs}")
    print(f"    IC:       {g}{_fmt(metrics.get('IC'))}{rs}")
    print(f"    Rank IC:  {g}{_fmt(metrics.get('Rank IC'))}{rs}")
    print(f"    RMSE:     {_fmt(metrics.get('RMSE'))}")
    print()

    # Factors used
    factor_names = sota.get("factor_names", [])
    print(f"  {b}Factors used ({len(factor_names)}):{rs}")
    for name in factor_names:
        print(f"    {c}- {name}{rs}")
    print()

    # Hypothesis
    hyp = sota.get("hypothesis", "")
    if hyp:
        print(f"  {b}Hypothesis:{rs}")
        for line in hyp.split("\n"):
            print(f"    {line}")
        print()

    # Feedback
    fb_reason = sota.get("feedback_reason", "")
    if fb_reason:
        print(f"  {b}Feedback Reason:{rs}")
        for line in fb_reason.split("\n"):
            print(f"    {line}")
        print()

    fb_obs = sota.get("feedback_observations", "")
    if fb_obs:
        print(f"  {b}Observations:{rs}")
        for line in str(fb_obs).split("\n"):
            print(f"    {line}")
        print()

    # File paths
    paths = sota.get("paths", {})
    if paths:
        print(f"  {b}File Locations:{rs}")

        # Experiment workspace
        ws_dir = paths.get("workspace_dir")
        if ws_dir:
            print(f"    {y}Experiment workspace:{rs}  {ws_dir}")

        # Aggregated factor file
        cf = paths.get("combined_factors_df.parquet")
        if cf:
            print(f"    {y}Combined factors:{rs}      {cf}")

        # Dataset
        ds = paths.get("crypto_dataset.parquet")
        if ds:
            print(f"    {y}Prepared dataset:{rs}      {ds}")

        # Config files
        for cfg in ["conf_baseline.yaml", "conf_combined_factors.yaml"]:
            cp = paths.get(cfg)
            if cp:
                print(f"    {y}Config ({cfg}):{rs}  {cp}")

        # Results
        for rfile in ["qlib_res.csv", "ret.pkl", "pred.pkl"]:
            rp = paths.get(rfile)
            if rp:
                print(f"    {y}Result ({rfile}):{rs}    {rp}")

        # Subdirectories
        for sdir in ["portfolio_analysis/", "signals/", "signal/", "mlflow/"]:
            sp = paths.get(sdir)
            if sp:
                print(f"    {y}{sdir}{rs}  {sp}")

        # Base features
        bf = paths.get("base_features")
        if bf:
            print(f"    {y}Base features:{rs}         {list(bf.keys())}")

        # Individual factor workspaces
        factor_wss = paths.get("factor_workspaces", [])
        if factor_wss:
            print()
            print(f"    {b}Individual factor files:{rs}")
            for fw in factor_wss:
                name = fw.get("factor_name", "?")
                fpy = fw.get("factor.py")
                fws = fw.get("workspace_dir")
                if fpy:
                    print(f"      {c}{name}{rs}: {fpy}")
                elif fws:
                    print(f"      {c}{name}{rs}: {fws}/factor.py")

        print()


def _print_table(factors: list[dict], sort_by: str, top: int | None,
                 sota_only: bool, show_code: bool, show_detail: bool,
                 no_color: bool):
    """Print the ranked factor summary table."""
    # Filter
    if sota_only:
        factors = [f for f in factors if f.get("is_sota")]

    # Sort
    reverse = sort_by in ("IC", "Rank IC")
    factors.sort(
        key=lambda f: f.get(sort_by) if f.get(sort_by) is not None else float("-inf"),
        reverse=reverse,
    )

    if top is not None:
        factors = factors[:top]

    if not factors:
        print("No factors found matching the criteria.")
        return

    sota_count = sum(1 for f in factors if f.get("is_sota"))

    b = "" if no_color else BOLD
    g = "" if no_color else GREEN
    c = "" if no_color else CYAN
    d = "" if no_color else DIM
    rs = "" if no_color else RESET

    print(f"{b}Factor Summary{rs} ({len(factors)} factors, {sota_count} SOTA)")
    print(f"Sorted by: {sort_by} ({'desc' if reverse else 'asc'})")
    print()

    # Table header
    hdr = f"{'Rank':>4}  {'Loop':>4}  {'Factor':<30}  {'IC':>10}  {'Rank IC':>10}  {'RMSE':>10}  {'SOTA':>4}"
    print(f"{b}{hdr}{rs}")
    print("-" * len(hdr) + "----")

    for rank, f in enumerate(factors, 1):
        ic = _fmt(f.get("IC"))
        ric = _fmt(f.get("Rank IC"))
        rmse = _fmt(f.get("RMSE"))
        name = f.get("factor_name", "?")[:30]
        loop = f.get("loop_id", "?")
        sota_marker = f"{g}***{rs}" if f.get("is_sota") else ""

        if not no_color and f.get("Rank IC") is not None and rank == 1 and reverse:
            ric = f"{g}{ric}{rs}"

        print(f"{rank:>4}  {loop:>4}  {name:<30}  {ic:>10}  {ric:>14}  {rmse:>10}  {sota_marker:>6}")

    print()

    # Detail mode: show paths and code for each factor
    if show_detail:
        for rank, f in enumerate(factors, 1):
            _print_factor_detail(f, rank, no_color)

    # Show code only (no paths)
    elif show_code:
        for rank, f in enumerate(factors, 1):
            code = f.get("factor_code")
            if code:
                print(f"{b}--- Rank {rank}: {f.get('factor_name', '?')} (Loop {f.get('loop_id', '?')}) ---{rs}")
                print(code)
                print()

    if not show_detail and not show_code:
        print(f"{d}Use --show-code to view factor implementations{rs}")
        print(f"{d}Use --detail to view paths, code, and metadata for each factor{rs}")
        print(f"{d}Use --export <path> to save full JSON report{rs}")


def _print_factor_detail(f: dict, rank: int, no_color: bool):
    """Print detailed info for a single factor."""
    b = "" if no_color else BOLD
    g = "" if no_color else GREEN
    c = "" if no_color else CYAN
    y = "" if no_color else YELLOW
    d = "" if no_color else DIM
    rs = "" if no_color else RESET

    name = f.get("factor_name", "?")
    loop = f.get("loop_id", "?")
    sota = "SOTA" if f.get("is_sota") else ""

    print(f"{b}--- Rank {rank}: {name} (Loop {loop}) {g}{sota}{rs} ---")
    print(f"  Description: {f.get('factor_description', '')}")
    formula = f.get("factor_formulation", "")
    if formula:
        print(f"  Formulation: {formula}")
    print(f"  IC: {_fmt(f.get('IC'))}  Rank IC: {_fmt(f.get('Rank IC'))}  RMSE: {_fmt(f.get('RMSE'))}")

    # factor.py path
    fpy = f.get("factor_py_path")
    if fpy:
        print(f"  {y}factor.py:{rs} {fpy}")

    # factor.py code
    code = f.get("factor_code")
    if code:
        print(f"  {c}Code:{rs}")
        for line in code.split("\n"):
            print(f"    {line}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and rank factors from an rdagent factor evolution run",
    )
    parser.add_argument("log_dir", type=Path, help="Path to the rdagent log directory")
    parser.add_argument("--sort-by", default="Rank IC",
                        choices=["IC", "Rank IC", "RMSE"],
                        help="Metric to sort by (default: Rank IC)")
    parser.add_argument("--top", type=int, default=None,
                        help="Show only top N factors")
    parser.add_argument("--sota-only", action="store_true",
                        help="Show only SOTA-beating factors")
    parser.add_argument("--show-code", action="store_true",
                        help="Print factor.py code for each factor")
    parser.add_argument("--detail", action="store_true",
                        help="Show detailed info: paths, code, metadata per factor")
    parser.add_argument("--export", type=Path, default=None,
                        help="Export results to a JSON file")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI color output")
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    if not log_dir.exists():
        print(f"Error: {log_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Try loading from factor_summary.json first (fast)
    factors, sota_detail = _load_from_json(log_dir)
    source = "factor_summary.json"

    # Fallback: load from session pickle
    if factors is None:
        print("Loading trace from session pickle (this may take a moment)...")
        trace = _load_trace_from_session(log_dir)
        if trace is None:
            print(f"Error: No session data found in {log_dir}", file=sys.stderr)
            sys.exit(1)
        factors, sota_detail = _extract_factors(trace)
        source = "session pickle"

    if not factors:
        print("No factors found in the log directory.")
        sys.exit(0)

    print(f"Loaded {len(factors)} factors from {source}")
    print()

    # Print SOTA detail section first
    if sota_detail:
        _print_sota_detail(sota_detail, args.no_color)

    # Export if requested
    if args.export:
        export_data = {
            "total_factors": len(factors),
            "sota_count": sum(1 for f in factors if f.get("is_sota")),
            "sort_by": args.sort_by,
            "sota_detail": sota_detail,
            "factors": factors,
        }
        with args.export.open("w") as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"Exported {len(factors)} factors to {args.export}")
        print()

    _print_table(factors, args.sort_by, args.top, args.sota_only,
                 args.show_code, args.detail, args.no_color)


if __name__ == "__main__":
    main()

"""
Binance factor workflow with session control
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import fire
import pandas as pd

from rdagent.app.binance_rd_loop.conf import BINANCE_FACTOR_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import CoderError, FactorEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.log.conf import LOG_SETTINGS


def _extract_workspace_paths(exp) -> dict:
    """Extract all relevant file paths from an experiment."""
    paths = {}
    ws = getattr(exp, "experiment_workspace", None)
    if ws is not None:
        ws_path = getattr(ws, "workspace_path", None)
        if ws_path is not None:
            paths["workspace_dir"] = str(ws_path)
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
            for subdir in ["portfolio_analysis", "signals", "signal", "mlflow"]:
                sd = ws_path / subdir
                if sd.exists() and sd.is_dir():
                    paths[f"{subdir}/"] = str(sd)

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
            factor_workspaces.append(fw_info)
    if factor_workspaces:
        paths["factor_workspaces"] = factor_workspaces

    base_features = getattr(exp, "base_features", {})
    base_feature_codes = getattr(exp, "base_feature_codes", {})
    if base_features:
        paths["base_features"] = base_features
    if base_feature_codes:
        paths["base_feature_codes"] = base_feature_codes

    return paths


class BinanceFactorRDLoop(RDLoop):
    skip_loop_error = (FactorEmptyError, CoderError)
    skip_loop_error_stepname = "feedback"

    def running(self, prev_out: dict[str, Any]):
        exp = self.runner.develop(prev_out["coding"])
        if exp is None:
            logger.error(f"Factor extraction failed.")
            raise FactorEmptyError("Factor extraction failed.")
        logger.log_object(exp, tag="runner result")
        return exp

    def record(self, prev_out: dict[str, Any]):
        super().record(prev_out)

        logger.log_object(self.trace, tag="trace")

        sota_hypo, sota_exp = self.trace.get_sota_hypothesis_and_experiment()
        if sota_exp is not None:
            logger.log_object(sota_exp, tag="SOTA experiment")

        self._write_factor_summary()

    def _write_factor_summary(self):
        """Write factor_summary.json to the log directory after each loop."""
        summary_path = Path(LOG_SETTINGS.trace_path) / "factor_summary.json"
        factors = []
        sota_detail = None

        for idx, (exp, fb) in enumerate(self.trace.hist):
            loop_id = self.trace.idx2loop_id.get(idx, idx)
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
            feedback_observations = getattr(fb, "observations", None) if fb is not None else None

            paths = _extract_workspace_paths(exp)

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

        factors.sort(key=lambda f: f.get("Rank IC") or float("-inf"), reverse=True)

        output = {
            "total_loops": len(self.trace.hist),
            "total_factors": len(factors),
            "sota_count": sum(1 for f in factors if f.get("is_sota")),
            "sota_detail": sota_detail,
            "factors": factors,
        }

        with summary_path.open("w") as f:
            json.dump(output, f, indent=2, default=str)


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: str | None = None,
    checkout: bool = True,
    checkout_path: Optional[str] = None,
    base_features_path: Optional[str] = None,
    **kwargs,
):
    """
    Auto R&D Evolving loop for Binance cryptocurrency factors.

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/binance_rd_loop/factor.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter

    """
    if not checkout_path is None:
        checkout = Path(checkout_path)

    if path is None:
        factor_loop = BinanceFactorRDLoop(BINANCE_FACTOR_PROP_SETTING)
    else:
        factor_loop = BinanceFactorRDLoop.load(path, checkout=checkout)

    factor_loop._init_base_features(base_features_path)
    if "user_interaction_queues" in kwargs and kwargs["user_interaction_queues"] is not None:
        factor_loop._set_interactor(*kwargs["user_interaction_queues"])
        factor_loop._interact_init_params()
    asyncio.run(factor_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)

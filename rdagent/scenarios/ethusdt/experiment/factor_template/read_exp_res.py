from pathlib import Path

import numpy as np
import pandas as pd
import qlib

qlib.init(provider_uri=".", region="cn")

from qlib.workflow import R


PRED_CANDIDATES = ["pred.pkl", "signals/pred.pkl", "signal/pred.pkl"]
LABEL_CANDIDATES = ["label.pkl", "signals/label.pkl", "signal/label.pkl"]


def _latest_recorder():
    latest = None
    for experiment_name in R.list_experiments():
        for recorder_id in R.list_recorders(experiment_name=experiment_name):
            if recorder_id is None:
                continue
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
            end_time = recorder.info.get("end_time")
            if end_time is None:
                continue
            if latest is None or end_time > latest.info.get("end_time"):
                latest = recorder
    return latest


def _load_first_existing(recorder, candidates):
    for candidate in candidates:
        try:
            return recorder.load_object(candidate)
        except Exception:
            continue
    return None


def _to_series(obj, preferred_names):
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        for name in preferred_names:
            if name in obj.columns:
                return obj[name]
        numeric_columns = obj.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            return obj[numeric_columns[0]]
        if obj.shape[1] > 0:
            return obj.iloc[:, 0]
    if isinstance(obj, np.ndarray):
        flat = obj.reshape(-1)
        return pd.Series(flat)
    return pd.Series(obj)


def _calculate_metrics(pred_obj, label_obj, recorder_metrics):
    pred = _to_series(pred_obj, ["score", "pred", "prediction"])
    label = _to_series(label_obj, ["label", "LABEL60", "target"])

    metrics = pd.Series(dtype="float64")

    if pred is not None and label is not None:
        aligned = pd.concat([pred.rename("pred"), label.rename("label")], axis=1).dropna()
    else:
        aligned = pd.DataFrame(columns=["pred", "label"])

    if not aligned.empty:
        metrics.loc["IC"] = aligned["pred"].corr(aligned["label"])
        metrics.loc["Rank IC"] = aligned["pred"].rank().corr(aligned["label"].rank())
        metrics.loc["RMSE"] = float(np.sqrt(((aligned["pred"] - aligned["label"]) ** 2).mean()))
    else:
        metrics.loc["IC"] = recorder_metrics.get("IC", np.nan)
        metrics.loc["Rank IC"] = recorder_metrics.get("Rank IC", np.nan)
        metrics.loc["RMSE"] = recorder_metrics.get("RMSE", np.nan)

    return metrics, aligned


def main():
    latest_recorder = _latest_recorder()
    if latest_recorder is None:
        raise RuntimeError("No completed Qlib recorder found.")

    recorder_metrics = pd.Series(latest_recorder.list_metrics(), dtype="float64")
    pred_obj = _load_first_existing(latest_recorder, PRED_CANDIDATES)
    label_obj = _load_first_existing(latest_recorder, LABEL_CANDIDATES)
    metrics, aligned = _calculate_metrics(pred_obj, label_obj, recorder_metrics)

    output_path = Path(__file__).resolve().parent / "qlib_res.csv"
    metrics.to_csv(output_path)

    ret_obj = aligned if not aligned.empty else metrics.to_frame(name="value")
    ret_obj.to_pickle(Path(__file__).resolve().parent / "ret.pkl")


if __name__ == "__main__":
    main()

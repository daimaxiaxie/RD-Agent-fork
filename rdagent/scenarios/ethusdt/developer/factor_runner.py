from pathlib import Path

import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(verbose=1)

from rdagent.app.ethusdt_rd_loop.conf import ETHUSDT_FACTOR_PROP_SETTING
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.runner import CachedRunner
from rdagent.core.exception import FactorEmptyError
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.ethusdt.experiment.factor_experiment import ETHUSDTFactorExperiment
from rdagent.scenarios.qlib.developer.utils import process_factor_data


class ETHUSDTFactorRunner(CachedRunner[ETHUSDTFactorExperiment]):
    def calculate_information_coefficient(
        self, concat_feature: pd.DataFrame, SOTA_feature_column_size: int, new_feature_columns_size: int
    ) -> pd.DataFrame:
        res = pd.Series(index=range(SOTA_feature_column_size * new_feature_columns_size))
        for col1 in range(SOTA_feature_column_size):
            for col2 in range(SOTA_feature_column_size, SOTA_feature_column_size + new_feature_columns_size):
                res.loc[col1 * new_feature_columns_size + col2 - SOTA_feature_column_size] = concat_feature.iloc[
                    :, col1
                ].corr(concat_feature.iloc[:, col2])
        return res

    def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
        concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
        ic_max = (
            concat_feature.groupby("datetime")
            .parallel_apply(
                lambda x: self.calculate_information_coefficient(x, SOTA_feature.shape[1], new_feature.shape[1])
            )
            .mean()
        )
        ic_max.index = pd.MultiIndex.from_product([range(SOTA_feature.shape[1]), range(new_feature.shape[1])])
        ic_max = ic_max.unstack().max(axis=0)
        return new_feature.iloc[:, ic_max[ic_max < 0.99].index]

    def _source_data_path(self) -> Path:
        return Path(FACTOR_COSTEER_SETTINGS.data_folder) / ETHUSDT_FACTOR_PROP_SETTING.source_data_file

    def _prepare_ohlcv_frame(self) -> pd.DataFrame:
        source_path = self._source_data_path()
        if not source_path.exists():
            raise FactorEmptyError(f"Source data file does not exist: {source_path}")

        source_df = pd.read_hdf(source_path, key="data")
        if source_df.empty:
            raise FactorEmptyError(f"Source data file is empty: {source_path}")

        if not isinstance(source_df.index, pd.MultiIndex):
            raise FactorEmptyError("Source data must use a MultiIndex with datetime and instrument levels.")

        normalized = source_df.copy()
        normalized.index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(normalized.index.get_level_values("datetime")),
                normalized.index.get_level_values("instrument").astype(str),
            ],
            names=["datetime", "instrument"],
        )
        normalized = normalized.sort_index()

        required_columns = ["$open", "$close", "$high", "$low", "$volume"]
        missing_columns = [col for col in required_columns if col not in normalized.columns]
        if missing_columns:
            raise FactorEmptyError(f"Source data is missing required columns: {missing_columns}")

        return normalized

    def _build_label_frame(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        horizon = ETHUSDT_FACTOR_PROP_SETTING.label_horizon_seconds
        close = ohlcv_df["$close"].unstack("instrument")
        future_return = close.shift(-horizon) / close - 1.0
        label = future_return.stack(dropna=False).to_frame(f"LABEL{horizon}")
        label.columns = pd.MultiIndex.from_product([["label"], label.columns])
        return label

    def _materialize_static_dataset(
        self,
        workspace_path: Path,
        combined_factors: pd.DataFrame | None = None,
    ) -> int:
        ohlcv_df = self._prepare_ohlcv_frame()

        base_feature = ohlcv_df[["$open", "$close", "$high", "$low", "$volume"]].copy()
        base_feature.columns = [col.replace("$", "") for col in base_feature.columns]
        base_feature.columns = pd.MultiIndex.from_product([["feature"], base_feature.columns])

        frames = [base_feature]
        feature_count = len(base_feature.columns)

        if combined_factors is not None and not combined_factors.empty:
            normalized_factors = combined_factors.sort_index()
            normalized_factors = normalized_factors.loc[:, ~normalized_factors.columns.duplicated(keep="last")]
            frames.append(normalized_factors)
            feature_count += len(normalized_factors.columns)

        label = self._build_label_frame(ohlcv_df)
        dataset = pd.concat(frames + [label], axis=1).sort_index()
        dataset = dataset.loc[:, ~dataset.columns.duplicated(keep="last")]
        dataset = dataset.dropna(subset=[("label", f"LABEL{ETHUSDT_FACTOR_PROP_SETTING.label_horizon_seconds}")])

        if dataset.empty:
            raise FactorEmptyError("Prepared cryptocurrency dataset is empty after aligning features and labels.")

        target_path = workspace_path / ETHUSDT_FACTOR_PROP_SETTING.prepared_dataset_file
        dataset.to_parquet(target_path, engine="pyarrow")
        logger.info(f"Prepared static cryptocurrency dataset at {target_path} with shape {dataset.shape}")
        return feature_count

    def _build_env(self, exp: ETHUSDTFactorExperiment, num_features: int) -> dict[str, str]:
        env_to_use = {
            "PYTHONPATH": "./",
            "instrument": "ETHUSDT",
            "train_start": ETHUSDT_FACTOR_PROP_SETTING.train_start,
            "train_end": ETHUSDT_FACTOR_PROP_SETTING.train_end,
            "valid_start": ETHUSDT_FACTOR_PROP_SETTING.valid_start,
            "valid_end": ETHUSDT_FACTOR_PROP_SETTING.valid_end,
            "test_start": ETHUSDT_FACTOR_PROP_SETTING.test_start,
            "label_horizon_seconds": str(ETHUSDT_FACTOR_PROP_SETTING.label_horizon_seconds),
            "source_data_file": ETHUSDT_FACTOR_PROP_SETTING.source_data_file,
            "prepared_dataset_file": ETHUSDT_FACTOR_PROP_SETTING.prepared_dataset_file,
            "num_features": str(num_features),
        }
        if ETHUSDT_FACTOR_PROP_SETTING.test_end is not None:
            env_to_use["test_end"] = ETHUSDT_FACTOR_PROP_SETTING.test_end
        return env_to_use

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: ETHUSDTFactorExperiment) -> ETHUSDTFactorExperiment:
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            logger.info("Baseline experiment execution ...")
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        combined_factors = None
        qlib_config_name = "conf_baseline.yaml"

        if exp.based_experiments:
            sota_factor = None
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, ETHUSDTFactorExperiment)
            ]
            if len(sota_factor_experiments_list) > 1:
                logger.info("SOTA factor processing ...")
                sota_factor = process_factor_data(sota_factor_experiments_list)

            logger.info("New factor processing ...")
            new_factors = process_factor_data(exp)
            if new_factors.empty:
                raise FactorEmptyError("Factors failed to run on the full sample, this round of experiment failed.")

            if sota_factor is not None and not sota_factor.empty:
                new_factors = self.deduplicate_new_factors(sota_factor, new_factors)
                if new_factors.empty:
                    raise FactorEmptyError(
                        "The factors generated in this round are highly similar to the previous factors. Please change the direction for creating new factors."
                    )
                merged_factors = pd.concat([sota_factor, new_factors], axis=1).dropna()
            else:
                merged_factors = new_factors

            merged_factors = merged_factors.sort_index()
            merged_factors = merged_factors.loc[:, ~merged_factors.columns.duplicated(keep="last")]
            merged_factors.columns = pd.MultiIndex.from_product([["feature"], merged_factors.columns])
            combined_factors = merged_factors
            combined_factors.to_parquet(
                exp.experiment_workspace.workspace_path / "combined_factors_df.parquet",
                engine="pyarrow",
            )
            qlib_config_name = "conf_combined_factors.yaml"
        elif exp.base_feature_codes:
            logger.info("Base feature processing ...")
            factors = process_factor_data(exp)
            factors = factors.sort_index()
            factors = factors.loc[:, ~factors.columns.duplicated(keep="last")]
            factors.columns = pd.MultiIndex.from_product([["feature"], factors.columns])
            combined_factors = factors
            combined_factors.to_parquet(
                exp.experiment_workspace.workspace_path / "combined_factors_df.parquet",
                engine="pyarrow",
            )
            qlib_config_name = "conf_combined_factors.yaml"

        num_features = self._materialize_static_dataset(exp.experiment_workspace.workspace_path, combined_factors)
        env_to_use = self._build_env(exp, num_features)

        logger.info("Experiment execution ...")
        result, stdout = exp.experiment_workspace.execute(
            qlib_config_name=qlib_config_name,
            run_env=env_to_use,
        )

        if result is None:
            logger.error(f"Failed to run this experiment, because {stdout}")
            raise FactorEmptyError(f"Failed to run this experiment, because {stdout}")

        exp.result = result
        exp.stdout = stdout
        return exp

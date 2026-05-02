from copy import deepcopy
from pathlib import Path

import pandas as pd

from rdagent.app.ethusdt_rd_loop.conf import ETHUSDT_FACTOR_PROP_SETTING
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS, get_factor_env
from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorFBWorkspace, FactorTask
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T


class ETHUSDTFactorExperiment(FactorExperiment[FactorTask, QlibFBWorkspace, FactorFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "factor_template")
        self.stdout = ""
        self.base_features: dict[str, str] = {}
        self.base_feature_codes: dict[str, str] = {}
        self.per_factor_ic: dict[str, dict[str, float]] = {}


class ETHUSDTFactorScenario(Scenario):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = self._infer_input_shape()
        self._background = deepcopy(
            T("scenarios.ethusdt.prompts:background").r(
                runtime_environment=self.get_runtime_environment(),
                label_horizon_seconds=ETHUSDT_FACTOR_PROP_SETTING.label_horizon_seconds,
            )
        )
        self._source_data = (
            f"Second-level cryptocurrency OHLCV data is stored in `{ETHUSDT_FACTOR_PROP_SETTING.source_data_file}` "
            "as a MultiIndex DataFrame `(datetime, instrument)` with columns `$open`, `$close`, `$high`, `$low`, `$volume`, `$factor`."
        )
        self._output_format = deepcopy(T("scenarios.ethusdt.prompts:output_format").r())
        self._interface = deepcopy(T("scenarios.ethusdt.prompts:interface").r())
        self._simulator = deepcopy(
            T("scenarios.ethusdt.prompts:simulator").r(label_horizon_seconds=ETHUSDT_FACTOR_PROP_SETTING.label_horizon_seconds)
        )
        self._rich_style_description = "Cryptocurrency second-level factor discovery scenario."
        self._experiment_setting = deepcopy(
            T("scenarios.ethusdt.prompts:experiment_setting").r(
                train_start=ETHUSDT_FACTOR_PROP_SETTING.train_start,
                train_end=ETHUSDT_FACTOR_PROP_SETTING.train_end,
                valid_start=ETHUSDT_FACTOR_PROP_SETTING.valid_start,
                valid_end=ETHUSDT_FACTOR_PROP_SETTING.valid_end,
                test_start=ETHUSDT_FACTOR_PROP_SETTING.test_start,
                test_end=ETHUSDT_FACTOR_PROP_SETTING.test_end,
                label_horizon_seconds=ETHUSDT_FACTOR_PROP_SETTING.label_horizon_seconds,
            )
        )

    def _infer_input_shape(self) -> tuple[int, int]:
        source_path = Path(FACTOR_COSTEER_SETTINGS.data_folder) / ETHUSDT_FACTOR_PROP_SETTING.source_data_file
        if not source_path.exists():
            return (0, 6)
        try:
            store = pd.HDFStore(str(source_path), mode="r")
            try:
                nrows = store.get_storer("data").nrows
                ncols = len(store.get_storer("data").attrs.non_index_axes[0][1])
            finally:
                store.close()
            return (nrows, ncols)
        except Exception:
            return (0, 6)

    @property
    def source_data_file(self) -> str:
        return ETHUSDT_FACTOR_PROP_SETTING.source_data_file

    @property
    def prepared_dataset_file(self) -> str:
        return ETHUSDT_FACTOR_PROP_SETTING.prepared_dataset_file

    @property
    def combined_factor_file(self) -> str:
        return "combined_factors_df.parquet"

    @property
    def baseline_config_name(self) -> str:
        return "conf_baseline.yaml"

    @property
    def combined_config_name(self) -> str:
        return "conf_combined_factors.yaml"

    @property
    def combined_model_config_name(self) -> str:
        return "conf_combined_factors_sota_model.yaml"

    @property
    def label_horizon_seconds(self) -> int:
        return ETHUSDT_FACTOR_PROP_SETTING.label_horizon_seconds

    @property
    def label_metric_names(self) -> list[str]:
        return ["IC", "Rank IC", "RMSE"]

    @property
    def instrument(self) -> str:
        return "ETHUSDT"

    @property
    def source_columns(self) -> list[str]:
        return ["$open", "$close", "$high", "$low", "$volume", "$factor"]

    @property
    def background(self) -> str:
        return self._background

    def get_source_data_desc(self, task: Task | None = None) -> str:
        return self._source_data

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        return self._experiment_setting

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        if simple_background:
            return f"""Background of the scenario:
{self.background}"""
        return f"""Background of the scenario:
{self.background}
The source data you can use:
{self.get_source_data_desc(task)}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your factor:
{self.simulator}
"""

    def get_runtime_environment(self):
        factor_env = get_factor_env()
        return get_runtime_environment_by_env(env=factor_env)

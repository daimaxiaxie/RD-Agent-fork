from copy import deepcopy
from pathlib import Path

from rdagent.app.ethusdt_rd_loop.conf import ETHUSDT_FACTOR_PROP_SETTING
from rdagent.components.coder.factor_coder.config import get_factor_env
from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T


class ETHUSDTFactorExperiment(FactorExperiment[FactorTask, QlibFBWorkspace, FactorFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(
            template_folder_path=Path(__file__).parent.parent.parent / "qlib" / "experiment" / "factor_template"
        )
        self.stdout = ""
        self.base_features: dict[str, str] = {}
        self.base_feature_codes: dict[str, str] = {}


class ETHUSDTFactorScenario(Scenario):
    def __init__(self) -> None:
        super().__init__()
        self._background = deepcopy(
            T(".prompts:background").r(
                runtime_environment=self.get_runtime_environment(),
            )
        )
        self._source_data = "ETH/USDT daily OHLCV data stored in `daily_pv.h5` as a multi-index DataFrame (datetime, instrument)."
        self._output_format = deepcopy(T(".prompts:output_format").r())
        self._interface = deepcopy(T(".prompts:interface").r())
        self._simulator = deepcopy(T(".prompts:simulator").r())
        self._rich_style_description = "ETH/USDT quantitative factor discovery scenario."
        self._experiment_setting = deepcopy(
            T(".prompts:experiment_setting").r(
                train_start=ETHUSDT_FACTOR_PROP_SETTING.train_start,
                train_end=ETHUSDT_FACTOR_PROP_SETTING.train_end,
                valid_start=ETHUSDT_FACTOR_PROP_SETTING.valid_start,
                valid_end=ETHUSDT_FACTOR_PROP_SETTING.valid_end,
                test_start=ETHUSDT_FACTOR_PROP_SETTING.test_start,
                test_end=ETHUSDT_FACTOR_PROP_SETTING.test_end,
            )
        )

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
        stdout = get_runtime_environment_by_env(env=factor_env)
        return stdout

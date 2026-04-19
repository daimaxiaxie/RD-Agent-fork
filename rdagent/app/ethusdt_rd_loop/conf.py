from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class ETHUSDTFactorPropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="ETHUSDT_FACTOR_", protected_namespaces=())

    scen: str = "rdagent.scenarios.ethusdt.experiment.factor_experiment.ETHUSDTFactorScenario"
    """Scenario class for ETH/USDT Factor"""

    hypothesis_gen: str = "rdagent.scenarios.ethusdt.proposal.factor_proposal.ETHUSDTFactorHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.ethusdt.proposal.factor_proposal.ETHUSDTFactorHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.ethusdt.developer.factor_coder.ETHUSDTFactorCoSTEER"
    """Coder class"""

    runner: str = "rdagent.scenarios.ethusdt.developer.factor_runner.ETHUSDTFactorRunner"
    """Runner class"""

    summarizer: str = "rdagent.scenarios.ethusdt.developer.feedback.ETHUSDTFactorExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""

    train_start: str = "2020-01-01"
    """Start date of the training segment"""

    train_end: str = "2023-12-31"
    """End date of the training segment"""

    valid_start: str = "2024-01-01"
    """Start date of the validation segment"""

    valid_end: str = "2024-06-30"
    """End date of the validation segment"""

    test_start: str = "2024-07-01"
    """Start date of the test / backtest segment"""

    test_end: Optional[str] = None
    """End date of the test / backtest segment"""


ETHUSDT_FACTOR_PROP_SETTING = ETHUSDTFactorPropSetting()

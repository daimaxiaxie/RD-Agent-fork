from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class BinanceFactorBasePropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="BINANCE_FACTOR_", protected_namespaces=())

    # 1) override base settings
    scen: str = "rdagent.scenarios.binance.experiment.factor_experiment.BinanceFactorScenario"
    """Scenario class for Binance Factor"""

    hypothesis_gen: str = "rdagent.scenarios.binance.proposal.factor_proposal.BinanceFactorHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.binance.proposal.factor_proposal.BinanceFactorHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.binance.developer.factor_coder.BinanceFactorCoSTEER"
    """Coder class"""

    runner: str = "rdagent.scenarios.binance.developer.factor_runner.BinanceFactorRunner"
    """Runner class"""

    summarizer: str = "rdagent.scenarios.binance.developer.feedback.BinanceFactorExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""

    train_start: str = "2008-01-01"
    """Start date of the training segment"""

    train_end: str = "2014-12-31"
    """End date of the training segment"""

    valid_start: str = "2015-01-01"
    """Start date of the validation segment"""

    valid_end: str = "2016-12-31"
    """End date of the validation segment"""

    test_start: str = "2017-01-01"
    """Start date of the test / backtest segment"""

    test_end: Optional[str] = "2020-08-01"
    """End date of the test / backtest segment"""


BINANCE_FACTOR_PROP_SETTING = BinanceFactorBasePropSetting()

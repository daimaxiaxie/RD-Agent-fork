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

    source_data_file: str = "crypto_1s.h5"
    """Scenario input file linked into each factor workspace"""

    prepared_dataset_file: str = "crypto_dataset.parquet"
    """Prepared training dataset written into the experiment workspace"""

    label_horizon_seconds: int = 60
    """Forward return horizon used as the default training label"""

    warmup_seconds: int = 3600
    """Seconds of data before train_start kept for factor warmup (e.g. rolling windows). Source data should cover this period."""

    evolving_n: int = 10
    """Number of evolutions"""

    train_start: str = "2025-11-01 00:00:00"
    """Start timestamp of the training segment"""

    train_end: str = "2026-01-31 23:59:59"
    """End timestamp of the training segment"""

    valid_start: str = "2026-02-01 00:00:00"
    """Start timestamp of the validation segment"""

    valid_end: str = "2026-02-28 23:59:59"
    """End timestamp of the validation segment"""

    test_start: str = "2026-03-01 00:00:00"
    """Start timestamp of the test segment"""

    test_end: Optional[str] = "2026-03-31 23:59:59"
    """End timestamp of the test segment"""

    trade_fee_rate: float = 0.001
    """One-way taker fee rate for perpetual futures (0.1% = 0.001). Round-trip cost is 2x."""


ETHUSDT_FACTOR_PROP_SETTING = ETHUSDTFactorPropSetting()

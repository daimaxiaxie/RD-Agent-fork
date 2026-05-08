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

    train_start: str = "2024-01-01"
    """Start date of the training segment"""

    train_end: str = "2024-09-30"
    """End date of the training segment"""

    valid_start: str = "2024-10-01"
    """Start date of the validation segment"""

    valid_end: str = "2024-12-31"
    """End date of the validation segment"""

    test_start: str = "2025-01-01"
    """Start date of the test / backtest segment"""

    test_end: Optional[str] = "2025-06-30"
    """End date of the test / backtest segment"""

    data_folder: str = "git_ignore_folder/binance_factor_implementation_source_data"
    """Path to the folder containing crypto hourly data for factor execution"""

    data_folder_debug: str = "git_ignore_folder/binance_factor_implementation_source_data_debug"
    """Path to the folder containing partial crypto hourly data (for debugging)"""

    qlib_provider_uri: str = "~/.qlib/qlib_data/crypto_data"
    """Qlib data provider URI for crypto hourly data"""

    market: str = "all"
    """Market name for Qlib (use 'all' for all instruments in crypto data)"""

    benchmark: str = "BTCUSDT"
    """Benchmark instrument for backtesting"""


BINANCE_FACTOR_PROP_SETTING = BinanceFactorBasePropSetting()

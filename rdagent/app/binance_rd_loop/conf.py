import re
from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting

# Mapping from qlib freq string to Binance kline interval
FREQ_TO_BINANCE_INTERVAL = {
    "60min": "1h",
    "240min": "4h",
    "480min": "8h",
    "720min": "12h",
}

# Mapping from qlib freq string to human-readable description
FREQ_TO_DESC = {
    "60min": "1-hourly",
    "240min": "4-hourly",
    "480min": "8-hourly",
    "720min": "12-hourly",
}


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
    """Path to the folder containing crypto data for factor execution"""

    data_folder_debug: str = "git_ignore_folder/binance_factor_implementation_source_data_debug"
    """Path to the folder containing partial crypto data (for debugging)"""

    qlib_provider_uri: str = "~/.qlib/qlib_data/crypto_data"
    """Qlib data provider URI for crypto data"""

    market: str = "all"
    """Market name for Qlib (use 'all' for all instruments in crypto data)"""

    benchmark: str = "BTCUSDT"
    """Benchmark instrument for backtesting"""

    freq: str = "60min"
    """Bar frequency for Qlib data and backtest (e.g., '60min' or '240min')"""

    @property
    def freq_minutes(self) -> int:
        """Parse freq string to total minutes (e.g., '60min' -> 60, '240min' -> 240)."""
        m = re.match(r"^(\d+)min$", self.freq)
        if not m:
            raise ValueError(f"Unsupported freq format: {self.freq}. Expected format like '60min' or '240min'.")
        return int(m.group(1))

    @property
    def freq_desc(self) -> str:
        """Human-readable description of the bar frequency (e.g., '1-hourly', '4-hourly')."""
        desc = FREQ_TO_DESC.get(self.freq)
        if desc is None:
            raise ValueError(f"No freq_desc mapping for freq={self.freq}. Add it to FREQ_TO_DESC.")
        return desc

    @property
    def ann_scaler(self) -> int:
        """Annualization scaler: number of bars per year."""
        return int(365.25 * 24 * 60 / self.freq_minutes)

    @property
    def binance_kline_interval(self) -> str:
        """Binance kline interval corresponding to the qlib freq."""
        interval = FREQ_TO_BINANCE_INTERVAL.get(self.freq)
        if interval is None:
            raise ValueError(f"No Binance kline interval mapping for freq={self.freq}. Add it to FREQ_TO_BINANCE_INTERVAL.")
        return interval

    @property
    def label_expression(self) -> str:
        """Qlib label expression for predicting next bar's return."""
        return "Ref($close, -1)/$close - 1"


BINANCE_FACTOR_PROP_SETTING = BinanceFactorBasePropSetting()

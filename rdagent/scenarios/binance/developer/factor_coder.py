from rdagent.app.binance_rd_loop.conf import BINANCE_FACTOR_PROP_SETTING
from rdagent.components.coder.factor_coder import FactorCoSTEER
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS

# Override data folder paths so factor execution uses binance-specific data
FACTOR_COSTEER_SETTINGS.data_folder = BINANCE_FACTOR_PROP_SETTING.data_folder
FACTOR_COSTEER_SETTINGS.data_folder_debug = BINANCE_FACTOR_PROP_SETTING.data_folder_debug

BinanceFactorCoSTEER = FactorCoSTEER

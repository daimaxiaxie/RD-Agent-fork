# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename         | Description                                                      |
| --------------   | -----------------------------------------------------------------|
| "hourly_pv.h5"  | Binance perpetual futures 1h OHLCV data (multi-coin).            |

# Data download

Run the download script to generate `hourly_pv_all.h5` and `hourly_pv_debug.h5`:

```bash
python rdagent/scenarios/binance/experiment/download_data.py --start 2021-01-01 --end 2024-12-31
```

# For different data, We have some basic knowledge for them

## Hourly price and volume data
$open: open price of the coin at that hour.
$close: close price of the coin at that hour.
$high: high price of the coin at that hour.
$low: low price of the coin at that hour.
$volume: volume of the coin at that hour.

Index: MultiIndex (datetime, instrument) where datetime is hourly and instrument is the symbol (e.g., "BTCUSDT").

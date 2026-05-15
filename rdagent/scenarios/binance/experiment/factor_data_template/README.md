# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename    | Description                                                    |
| -------------- | -----------------------------------------------------------------|
| "pv.h5"     | Binance perpetual futures OHLCV data (multi-coin).             |

# Data download

Run the download script to generate `pv_all.h5` and `pv_debug.h5`:

```bash
python rdagent/scenarios/binance/experiment/download_data.py --start 2024-01-01 --end 2025-06-30
```

# For different data, We have some basic knowledge for them

## Price and volume data
$open: open price of the coin at that bar.
$close: close price of the coin at that bar.
$high: high price of the coin at that bar.
$low: low price of the coin at that bar.
$volume: volume of the coin at that bar.

Index: MultiIndex (datetime, instrument) where datetime is the bar timestamp and instrument is the symbol (e.g., "BTCUSDT").

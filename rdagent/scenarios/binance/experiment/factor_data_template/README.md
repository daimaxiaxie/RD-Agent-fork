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
| "pv.h5"     | Binance perpetual futures OHLCV + metrics data (multi-coin).   |

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

## Open interest
$oi: total open interest in USDT value for the symbol at that bar. Higher OI indicates more capital deployed in the market.

## Top trader long/short ratio (by position)
$top_ls_pos: the ratio of top traders' long vs short positions by position size. Values > 1 indicate top traders are net long; < 1 indicate net short.

## Top trader long/short ratio (by account count)
$top_ls_acc: the ratio of top traders' long vs short accounts by account count. Values > 1 indicate more top trader accounts are long; < 1 indicate more are short.

## Global long/short account ratio
$global_ls: the ratio of all traders' long vs short accounts. Values > 1 indicate the majority of accounts hold long positions; < 1 indicate the majority hold short.

## Taker buy/sell volume ratio
$taker_ls: the ratio of taker buy volume vs taker sell volume. Values > 1 indicate aggressive buying (takers buying); < 1 indicate aggressive selling (takers selling).

## Funding rate
$funding_rate: the perpetual futures funding rate. Positive values mean longs pay shorts (bullish sentiment costs); negative values mean shorts pay longs (bearish sentiment costs). Updated every 8 hours, forward-filled to bar frequency.

Index: MultiIndex (datetime, instrument) where datetime is the bar timestamp and instrument is the symbol (e.g., "BTCUSDT").

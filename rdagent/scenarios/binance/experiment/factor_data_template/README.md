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
$volume: volume (base asset) of the coin at that bar.
$quote_volume: volume in quote asset (USDT) at that bar.
$count: number of trades at that bar.

## Taker buy/sell volume
$taker_buy_vol: taker buy volume (base asset) — volume from aggressive buy orders.
$taker_buy_quote_vol: taker buy volume in quote asset (USDT) — notional from aggressive buy orders.
$taker_sell_vol: taker sell volume (base asset) — volume from aggressive sell orders.
$taker_sell_quote_vol: taker sell volume in quote asset (USDT) — notional from aggressive sell orders.

## Open interest
$oi: total open interest in contracts (number of outstanding positions) for the symbol at that bar.
$oi_value: total open interest in USDT value for the symbol at that bar. Higher OI indicates more capital deployed in the market.

## Top trader long/short ratio (by position)
$top_ls_pos: the ratio of top traders' long vs short positions by position size. Values > 1 indicate top traders are net long; < 1 indicate net short.

## Top trader long/short ratio (by account count)
$top_ls_acc: the ratio of top traders' long vs short accounts by account count. Values > 1 indicate more top trader accounts are long; < 1 indicate more are short.

## Global long/short account ratio
$global_ls: the ratio of all traders' long vs short accounts. Values > 1 indicate the majority of accounts hold long positions; < 1 indicate the majority hold short.

## Taker buy/sell volume ratio
$taker_ls: the ratio of taker buy volume vs taker sell volume. Values > 1 indicate aggressive buying (takers buying); < 1 indicate aggressive selling (takers selling).

## Mark price
$mark_close: the mark price (fair price used by the exchange for liquidation calculations) at the end of the bar. This is different from $close (last trade price). The mark price is computed from the index price and recent premium median, and represents the exchange's fair valuation.

## Basis (premium/discount signal)
$basis: (mark_close - close) / close. This measures the premium or discount of the futures contract relative to the mark price. Positive basis means the futures are trading above the fair price (bullish premium); negative means below (bearish discount). This is a crypto-specific signal derived from the futures market structure.

## VWAP (volume-weighted average price)
$vwap: quote_volume / volume. The volume-weighted average price for the bar, computed from the total traded notional divided by total traded base units. More robust than simple average price, often used as a benchmark for execution quality.

## Funding rate
$funding_rate: the perpetual futures funding rate. Positive values mean longs pay shorts (bullish sentiment costs); negative values mean shorts pay longs (bearish sentiment costs). Updated every 8 hours, forward-filled to bar frequency.

Index: MultiIndex (datetime, instrument) where datetime is the bar timestamp and instrument is the symbol (e.g., "BTCUSDT").

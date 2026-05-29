#!/usr/bin/env python3
"""
Download multi-coin klines + crypto-specific metrics (open interest, long/short
ratios, taker ratio) and funding rate from Binance data.binance.vision
(perpetual futures only) and save as HDF5 for the binance_factor scenario.

Output files:
  - pv_all.h5       full dataset (key="data")
  - pv_debug.h5     subset: first 5 coins, 6 months around midpoint

Index:  MultiIndex (datetime, instrument)
Columns: $open, $high, $low, $close, $volume, $quote_volume, $count,
         $taker_buy_vol, $taker_buy_quote_vol, $taker_sell_vol, $taker_sell_quote_vol,
         $vwap,
         $oi, $oi_value, $top_ls_pos, $top_ls_acc, $global_ls, $taker_ls, $funding_rate,
         $mark_close, $basis

Usage:
    python download_data.py                          # default: 1h klines + metrics
    python download_data.py --interval 4h            # 4h klines + metrics
    python download_data.py --start 2021-01-01 --end 2024-12-31
    python download_data.py --no-metrics              # klines only (old behavior)
    # After download, generate Qlib binary data separately:
    python generate_qlib_data.py
"""

import argparse
import logging
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://data.binance.vision"
OUTPUT_DIR = Path(__file__).resolve().parent / "factor_data_template"

log = logging.getLogger("download_binance_data")

SYMBOLS = [
    # 压舱石
    "BTCUSDT", "ETHUSDT", "BNBUSDT",
    # 强动能
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "DOTUSDT", "LINKUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT",
    "TRXUSDT", "TONUSDT", "AAVEUSDT",
    "CRVUSDT", "SNXUSDT", "GRTUSDT", "FILUSDT", "STXUSDT",
    "IMXUSDT", "ETCUSDT", "BCHUSDT", "SANDUSDT", "ALGOUSDT",
    "ICPUSDT", "FLOWUSDT", "THETAUSDT", "AXSUSDT",
    "GALAUSDT", "1000PEPEUSDT", "PENDLEUSDT", "LDOUSDT", "ENSUSDT",
    "PYTHUSDT", "IOTAUSDT", "AXLUSDT", "EGLDUSDT", "CKBUSDT",
    "AEROUSDT", "KAIAUSDT", "DRIFTUSDT",
    # 增强器
    "OPUSDT", "ARBUSDT", "APTUSDT", "NEARUSDT", "INJUSDT",
    "TIAUSDT", "SUIUSDT", "FETUSDT", "WLDUSDT",
    "CFXUSDT", "MASKUSDT", "DYDXUSDT", "1000FLOKIUSDT", "ONDOUSDT",
    "BOMEUSDT", "MEMEUSDT", "COMPUSDT", "ENAUSDT",
    "GOATUSDT", "NEIROUSDT", "JUPUSDT", "WIFUSDT",
    # 新增精选
    "1000SHIBUSDT", "RENDERUSDT", "EIGENUSDT", "KASUSDT",
    "TAOUSDT", "ORDIUSDT", "MKRUSDT", "HBARUSDT",
    "1000BONKUSDT", "ZECUSDT", "BLURUSDT", "SSVUSDT",
    "RUNEUSDT", "VETUSDT", "ACTUSDT", "POPCATUSDT",
    "ZROUSDT", "POLUSDT", "APEUSDT",
]

METRICS_COL_MAP = {
    "sum_open_interest": "$oi",
    "sum_open_interest_value": "$oi_value",
    "sum_toptrader_long_short_ratio": "$top_ls_pos",
    "count_toptrader_long_short_ratio": "$top_ls_acc",
    "count_long_short_ratio": "$global_ls",
    "sum_taker_long_short_vol_ratio": "$taker_ls",
}

# Interval to resample rule mapping for metrics (5min source → target)
INTERVAL_RESAMPLE = {"1h": "1h", "4h": "4h", "1d": "1D"}


def _download_zip(url: str, timeout: int = 60, retries: int = 2) -> pd.DataFrame | None:
    data = None
    for attempt in range(retries + 1):
        try:
            resp = urllib.request.urlopen(url, timeout=timeout)
            data = resp.read()
            break
        except urllib.error.HTTPError:
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries:
                import time
                time.sleep(1 * (attempt + 1))
            else:
                log.warning("Request failed after %d retries: %s", retries, e)
                return None

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "data.zip"
        zip_path.write_bytes(data)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return None
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
        except zipfile.BadZipFile:
            log.warning("Bad zip: %s", url)
            return None
    return df


def _download_klines(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, interval: str) -> pd.DataFrame | None:
    """Download klines for one symbol at the given interval. Returns None if any month is missing."""
    chunks = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        filename = f"{symbol}-{interval}-{current.strftime('%Y-%m')}.zip"
        url = f"{BASE_URL}/data/futures/um/monthly/klines/{symbol}/{interval}/{filename}"
        df = _download_zip(url)
        if df is None:
            log.warning("  %s %s: not available, skipping symbol", symbol, current.strftime("%Y-%m"))
            return None
        chunks.append(df)
        log.info("  %s %s: %d rows", symbol, current.strftime("%Y-%m"), len(df))
        current += pd.DateOffset(months=1)

    full = pd.concat(chunks, ignore_index=True)
    full.columns = [f"col_{i}" for i in range(len(full.columns))]
    full["datetime"] = pd.to_datetime(full["col_0"], unit="ms", utc=True).dt.tz_localize(None)
    full = full.rename(columns={
        "col_1": "$open", "col_2": "$high", "col_3": "$low",
        "col_4": "$close", "col_5": "$volume",
        "col_7": "$quote_volume", "col_8": "$count",
        "col_9": "$taker_buy_vol", "col_10": "$taker_buy_quote_vol",
    })
    full = full.set_index("datetime").sort_index()
    full = full[~full.index.duplicated(keep="first")]
    keep = ["$open", "$high", "$low", "$close", "$volume",
            "$quote_volume", "$count", "$taker_buy_vol", "$taker_buy_quote_vol"]
    full = full[keep].astype(np.float64)
    # Derive taker sell side from total - taker buy
    full["$taker_sell_vol"] = full["$volume"] - full["$taker_buy_vol"]
    full["$taker_sell_quote_vol"] = full["$quote_volume"] - full["$taker_buy_quote_vol"]
    # VWAP = quote_volume / volume (volume-weighted average price, no look-ahead)
    full["$vwap"] = full["$quote_volume"] / full["$volume"].replace(0, np.nan)
    return full.loc[start_dt:end_dt]


def _download_metrics(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, interval: str) -> pd.DataFrame | None:
    """Download daily metrics zips (5-min granularity), resample to target interval.

    Uses concurrent downloads to speed up the per-day requests.
    Returns DataFrame with columns: $oi, $oi_value, $top_ls_pos, $top_ls_acc,
    $global_ls, $taker_ls indexed by datetime.
    """
    rule = INTERVAL_RESAMPLE.get(interval, "1h")
    current = start_dt.normalize()
    end_norm = end_dt.normalize() + pd.Timedelta(days=1)

    # Build all (date, url) pairs
    date_urls = []
    while current < end_norm:
        date_str = current.strftime("%Y-%m-%d")
        url = f"{BASE_URL}/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{date_str}.zip"
        date_urls.append((date_str, url))
        current += pd.Timedelta(days=1)

    total_days = len(date_urls)

    # Concurrent download of daily zips
    chunks = []
    missing_days = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_to_date = {
            pool.submit(_download_zip, url): date_str
            for date_str, url in date_urls
        }
        done_count = 0
        for future in as_completed(future_to_date):
            done_count += 1
            try:
                df = future.result()
            except Exception:
                df = None
            if df is None:
                missing_days += 1
            else:
                chunks.append(df)
            if done_count % 200 == 0 or done_count == total_days:
                log.info("  %s metrics: %d/%d days downloaded", symbol, done_count, total_days)

    if not chunks:
        log.warning("  %s metrics: no data available", symbol)
        return None

    if missing_days > 0:
        log.info("  %s metrics: %d days missing", symbol, missing_days)

    full = pd.concat(chunks, ignore_index=True)
    full["datetime"] = pd.to_datetime(full["create_time"], utc=True).dt.tz_localize(None)
    full = full.set_index("datetime").sort_index()
    full = full[~full.index.duplicated(keep="first")]

    # Select and rename columns
    available = {k: v for k, v in METRICS_COL_MAP.items() if k in full.columns}
    result = full[list(available.keys())].rename(columns=available).astype(np.float64)

    # Resample 5min → target interval with semantic-correct aggregation
    # - OI / OI_value: snapshot at period end (instantaneous stock variable)
    # - Ratios (top_ls_pos, top_ls_acc, global_ls, taker_ls): period average
    #   (ratios are flow-like over the window; last() is too noisy)
    agg_map = {}
    for col in result.columns:
        if col in ("$oi", "$oi_value"):
            agg_map[col] = "last"
        else:
            agg_map[col] = "mean"
    result = result.resample(rule).agg(agg_map)

    return result.loc[start_dt:end_dt]


def _download_funding_rate(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, interval: str) -> pd.DataFrame | None:
    """Download monthly funding rate zips (8h granularity), forward-fill to target interval.

    Returns DataFrame with column: $funding_rate, indexed by datetime.
    """
    rule = INTERVAL_RESAMPLE.get(interval, "1h")
    chunks = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        filename = f"{symbol}-fundingRate-{current.strftime('%Y-%m')}.zip"
        url = f"{BASE_URL}/data/futures/um/monthly/fundingRate/{symbol}/{filename}"
        df = _download_zip(url)
        if df is None:
            log.warning("  %s fundingRate %s: not available", symbol, current.strftime("%Y-%m"))
            current += pd.DateOffset(months=1)
            continue
        chunks.append(df)
        current += pd.DateOffset(months=1)

    if not chunks:
        log.warning("  %s fundingRate: no data available", symbol)
        return None

    full = pd.concat(chunks, ignore_index=True)
    full["datetime"] = pd.to_datetime(full["calc_time"], unit="ms", utc=True).dt.tz_localize(None)
    full = full.set_index("datetime").sort_index()
    full = full[~full.index.duplicated(keep="first")]

    result = full[["last_funding_rate"]].rename(columns={"last_funding_rate": "$funding_rate"}).astype(np.float64)

    # Funding rate is 8h; reindex to target interval and forward-fill
    target_idx = pd.date_range(start_dt, end_dt, freq=rule)
    result = result.reindex(result.index.union(target_idx)).sort_index()
    result = result.ffill()
    result = result.reindex(target_idx)
    result.index.name = "datetime"

    return result.loc[start_dt:end_dt]


def _download_mark_price(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, interval: str) -> pd.DataFrame | None:
    """Download monthly mark price klines (same interval as regular klines).

    Mark price is the exchange's fair price used for liquidation calculations,
    based on the index price and a decaying median of recent premium.
    Returns DataFrame with columns: $mark_close, $basis.
    No look-ahead: mark price is the exchange's published fair price at bar close.
    """
    chunks = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        filename = f"{symbol}-{interval}-{current.strftime('%Y-%m')}.zip"
        url = f"{BASE_URL}/data/futures/um/monthly/markPriceKlines/{symbol}/{interval}/{filename}"
        df = _download_zip(url)
        if df is None:
            log.warning("  %s markPrice %s: not available, skipping", symbol, current.strftime("%Y-%m"))
            current += pd.DateOffset(months=1)
            continue
        chunks.append(df)
        current += pd.DateOffset(months=1)

    if not chunks:
        log.warning("  %s markPrice: no data available", symbol)
        return None

    full = pd.concat(chunks, ignore_index=True)
    full.columns = [f"col_{i}" for i in range(len(full.columns))]
    full["datetime"] = pd.to_datetime(full["col_0"], unit="ms", utc=True).dt.tz_localize(None)
    full = full.rename(columns={"col_4": "$mark_close"})
    full = full.set_index("datetime").sort_index()
    full = full[~full.index.duplicated(keep="first")]
    result = full[["$mark_close"]].astype(np.float64)

    return result.loc[start_dt:end_dt]


def main():
    parser = argparse.ArgumentParser(description="Download Binance perpetual futures klines + metrics -> pv.h5")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--interval", default="1h", choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                        help="Binance kline interval (default: 1h)")
    parser.add_argument("--output", help="Output path (default: factor_data_template/pv_all.h5)")
    parser.add_argument("--debug-output", help="Debug path (default: factor_data_template/pv_debug.h5)")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Skip metrics/funding rate download (klines only)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)
    include_metrics = not args.no_metrics

    log.info("Downloading %d symbols at %s interval with 4 threads", len(SYMBOLS), args.interval)

    # --- Step 1: Download klines (as before) ---
    all_frames = []
    symbol_kline_idx = {}  # symbol -> kline datetime index for alignment
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_download_klines, s, start_dt, end_dt, args.interval): s for s in SYMBOLS}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
            except Exception as e:
                log.warning("Error downloading %s klines: %s", symbol, e)
                continue
            if df is None or df.empty:
                log.warning("No kline data for %s, skipping.", symbol)
                continue
            log.info("Finished klines %s", symbol)
            df = df.copy()
            df["instrument"] = symbol
            df = df.set_index([df.index, "instrument"])
            df.index.names = ["datetime", "instrument"]
            symbol_kline_idx[symbol] = df.index.get_level_values("datetime")
            all_frames.append(df)

    if not all_frames:
        log.error("No data downloaded.")
        raise SystemExit(1)

    # --- Step 2: Download metrics + funding rate, merge per symbol ---
    if include_metrics and args.interval in INTERVAL_RESAMPLE:
        log.info("Downloading metrics and funding rate for %d symbols...", len(symbol_kline_idx))

        def _download_and_merge(symbol: str, kline_df: pd.DataFrame) -> pd.DataFrame:
            kline_idx = kline_df.index.get_level_values("datetime")

            # Download metrics
            metrics_df = _download_metrics(symbol, start_dt, end_dt, args.interval)

            # Download funding rate
            funding_df = _download_funding_rate(symbol, start_dt, end_dt, args.interval)

            # Download mark price
            mark_df = _download_mark_price(symbol, start_dt, end_dt, args.interval)

            # Align to kline datetime index
            extra_cols = []
            if metrics_df is not None and not metrics_df.empty:
                metrics_aligned = metrics_df.reindex(kline_idx)
                extra_cols.append(metrics_aligned)
            if funding_df is not None and not funding_df.empty:
                funding_aligned = funding_df.reindex(kline_idx)
                extra_cols.append(funding_aligned)
            if mark_df is not None and not mark_df.empty:
                mark_aligned = mark_df.reindex(kline_idx)
                extra_cols.append(mark_aligned)

            if extra_cols:
                extras = pd.concat(extra_cols, axis=1)
                # Derive $basis = ($mark_close - $close) / $close (premium/discount signal)
                # Both kline_df and extras share the same MultiIndex (datetime, instrument)
                # for a single symbol, so .values alignment is safe here.
                if "$mark_close" in extras.columns:
                    close_vals = kline_df["$close"].values
                    mark_vals = extras["$mark_close"].values
                    extras["$basis"] = (mark_vals - close_vals) / np.where(close_vals != 0, close_vals, np.nan)
                extras["instrument"] = symbol
                extras = extras.set_index([extras.index, "instrument"])
                extras.index.names = ["datetime", "instrument"]
                kline_df = pd.concat([kline_df, extras], axis=1)

            return kline_df

        merged_frames = []
        total_symbols = len(symbol_kline_idx)
        finished_count = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_download_and_merge, symbol, df): symbol
                for symbol, df in zip(symbol_kline_idx.keys(), all_frames)
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    merged = future.result()
                    merged_frames.append(merged)
                    finished_count += 1
                    log.info("Finished metrics %s (%d/%d)", symbol, finished_count, total_symbols)
                except Exception as e:
                    log.warning("Error downloading metrics for %s: %s", symbol, e)
                    # Fall back to kline-only for this symbol
                    idx = list(symbol_kline_idx.keys()).index(symbol)
                    merged_frames.append(all_frames[idx])
                    finished_count += 1

        all_frames = merged_frames
    elif include_metrics:
        log.warning("Metrics download not supported for interval=%s (only 1h/4h/1d)", args.interval)

    combined = pd.concat(all_frames).sort_index().dropna(how="all")

    log.info("Combined: %d rows, %d instruments, %s -> %s",
             len(combined),
             combined.index.get_level_values("instrument").nunique(),
             combined.index.get_level_values("datetime").min(),
             combined.index.get_level_values("datetime").max())
    log.info("Columns: %s", list(combined.columns))

    # Save full (HDF5/PyTables does not support category in MultiIndex)
    out_path = Path(args.output) if args.output else OUTPUT_DIR / "pv_all.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_hdf(out_path, key="data", mode="w")
    log.info("Saved to %s", out_path)

    # Save debug (5 coins, 6 months around midpoint)
    debug_path = Path(args.debug_output) if args.debug_output else OUTPUT_DIR / "pv_debug.h5"
    debug_coins = combined.index.get_level_values("instrument").unique()[:5]
    dt_min = combined.index.get_level_values("datetime").min()
    dt_max = combined.index.get_level_values("datetime").max()
    mid = dt_min + (dt_max - dt_min) / 2
    debug_df = combined.loc[
        combined.index.get_level_values("instrument").isin(debug_coins) &
        (combined.index.get_level_values("datetime") >= mid - pd.Timedelta(days=90)) &
        (combined.index.get_level_values("datetime") <= mid + pd.Timedelta(days=90))
    ]
    debug_df.to_hdf(debug_path, key="data", mode="w")
    log.info("Saved debug (%d rows, %d coins) to %s", len(debug_df), len(debug_coins), debug_path)

    log.info("Download complete. To generate Qlib binary data, run:\n"
             "  python rdagent/scenarios/binance/experiment/generate_qlib_data.py --freq <freq>")


if __name__ == "__main__":
    main()

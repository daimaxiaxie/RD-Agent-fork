#!/usr/bin/env python3
"""
Download Binance klines via CCXT API and convert to RD-Agent h5.

Supports spot, USDT-M futures (um), COIN-M futures (cm).
Minimum timeframe is 1m (API does not support 1s). For 1s data, use download_binance_public.py.

Features:
- Automatic rate limiting and retry on Binance API errors
- Resumes partial downloads via per-chunk caching

Usage:
    # USDT-M futures 1m klines
    python scripts/download_binance_api.py --symbol ETHUSDT --timeframe 1m --type um

    # Spot 1h klines
    python scripts/download_binance_api.py --symbol ETHUSDT --timeframe 1h --type spot

    # Custom date range
    python scripts/download_binance_api.py --start 2025-01-01 --end 2025-12-31
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("git_ignore_folder/factor_implementation_source_data")
OUTPUT_FILE = "crypto_1m.h5"
HDF_KEY = "data"
CHUNK_CACHE_DIR = Path("git_ignore_folder/.binance_api_klines_cache")

BINANCE_RATE_LIMIT_MS = 200
BINANCE_KLINE_LIMIT = 1000

_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
RDAGENT_COLS = ["$open", "$close", "$high", "$low", "$volume", "$factor"]

log = logging.getLogger("download_binance_api")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _symbol_to_instrument(symbol: str) -> str:
    return symbol.replace("/", "")


def _to_ccxt_symbol(symbol: str) -> str:
    """Convert plain symbol (ETHUSDT) to CCXT format (ETH/USDT)."""
    if "/" in symbol:
        return symbol
    # Try common quote currencies
    for quote in ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"]:
        if symbol.endswith(quote):
            return symbol[:-len(quote)] + "/" + quote
    return symbol


def _save_hdf(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(path, key=HDF_KEY, mode="w")
    log.info("Saved %s rows to %s", len(df), path)


def _to_rdagent_format(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    out = df.rename(columns={
        "open": "$open", "high": "$high", "low": "$low",
        "close": "$close", "volume": "$volume",
    })
    out["$factor"] = 1.0
    out.index.name = "datetime"
    out["instrument"] = instrument
    out = out.reset_index().set_index(["datetime", "instrument"]).sort_index()
    return out[RDAGENT_COLS]


# ── Download ──────────────────────────────────────────────────────────────────
def download_api_klines(
    symbol: str,
    timeframe: str = "1m",
    trading_type: str = "um",
    start: str = "2025-10-31",
    end: str = "2026-03-31",
) -> pd.DataFrame:
    import ccxt

    exchange_cls = {
        "spot": ccxt.binance,
        "um": ccxt.binanceusdm,
        "cm": ccxt.binancecoinm,
    }[trading_type]
    exchange = exchange_cls({"enableRateLimit": True})

    ccxt_symbol = _to_ccxt_symbol(symbol)
    instrument = _symbol_to_instrument(symbol)

    CHUNK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    start_ms = exchange.parse8601(start + "T00:00:00Z")
    end_ms = exchange.parse8601(end + "T23:59:59Z")
    tf_ms = exchange.parse_timeframe(timeframe) * 1000

    total_candles = (end_ms - start_ms) // tf_ms
    estimated_requests = (total_candles // BINANCE_KLINE_LIMIT) + 1
    log.info("Downloading %s %s klines (%s): %s -> %s (~%d candles, ~%d API requests)",
             ccxt_symbol, timeframe, trading_type, start, end, total_candles, estimated_requests)

    frames = []
    cursor_ms = start_ms
    req_count = 0

    while cursor_ms < end_ms:
        chunk_file = CHUNK_CACHE_DIR / f"{instrument}_{timeframe}_{cursor_ms}.parquet"
        if chunk_file.exists():
            chunk = pd.read_parquet(chunk_file)
            if not chunk.empty:
                frames.append(chunk)
                cursor_ms = int(chunk["datetime"].max().timestamp() * 1000) + tf_ms
                continue

        ohlcv = None
        for attempt in range(3):
            try:
                ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=cursor_ms, limit=BINANCE_KLINE_LIMIT)
                break
            except ccxt.RateLimitExceeded:
                wait = 10 + attempt * 5
                log.warning("Rate limited, waiting %ds...", wait)
                time.sleep(wait)
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                wait = 2 + attempt * 2
                log.warning("Exchange error (attempt %d): %s — retrying in %ds", attempt + 1, e, wait)
                time.sleep(wait)

        if ohlcv is None:
            log.error("Failed after 3 retries for since=%s", cursor_ms)
            break
        if len(ohlcv) == 0:
            break

        chunk = pd.DataFrame(ohlcv, columns=_COLUMNS)
        chunk["datetime"] = pd.to_datetime(chunk["datetime"], unit="ms", utc=True)
        chunk = chunk.set_index("datetime")

        chunk.reset_index().to_parquet(chunk_file, index=False)
        frames.append(chunk)

        cursor_ms = int(ohlcv[-1][0]) + tf_ms
        req_count += 1

        if req_count % 50 == 0:
            pct = min(100, (cursor_ms - start_ms) / (end_ms - start_ms) * 100)
            log.info("Progress: %.0f%% (%d requests, %d candles)", pct, req_count, sum(len(f) for f in frames))

        time.sleep(BINANCE_RATE_LIMIT_MS / 1000)

    if not frames:
        log.error("No data downloaded.")
        sys.exit(1)

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    log.info("Downloaded %s candles (%s → %s)", len(df), df.index.min(), df.index.max())
    return df


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download Binance klines via API → RD-Agent h5",
    )
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--timeframe", default="1m",
                        help="Kline interval (min 1m, use download_binance_public.py for 1s)")
    parser.add_argument("--type", choices=["spot", "um", "cm"], default="um",
                        help="Market type (default: um = USDT-M futures)")
    parser.add_argument("--start", default="2025-10-31")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--output", help="Output .h5 path (default: auto)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.timeframe == "1s":
        log.error("API does not support 1s. Use download_binance_public.py instead.")
        sys.exit(1)

    instrument = _symbol_to_instrument(args.symbol)
    out_path = Path(args.output) if args.output else OUTPUT_DIR / OUTPUT_FILE

    t0 = time.time()
    raw = download_api_klines(
        symbol=args.symbol,
        timeframe=args.timeframe,
        trading_type=args.type,
        start=args.start,
        end=args.end,
    )
    out = _to_rdagent_format(raw, instrument)
    _save_hdf(out, out_path)

    log.info("Completed in %.0fs", time.time() - t0)
    log.info("  Range: %s → %s", out.index.get_level_values("datetime").min(),
             out.index.get_level_values("datetime").max())
    log.info("  Rows:  %d", len(out))


if __name__ == "__main__":
    main()

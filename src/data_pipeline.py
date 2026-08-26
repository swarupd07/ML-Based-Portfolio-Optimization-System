# Data pipeline: downloading historical prices and engineer features

"""
------------------------------------------------------------------------------------
                                Design notes by Swarup 
------------------------------------------------------------------------------------

- All feature engineering here is causal (uses only past data at each row) so
  that later, when we slice by date for walk-forward validation, we don't
  need to re-derive anything (the leakage-avoidance is baked in upstream)

- Missing data is forward filled per ticker, never cross sectionally, to avoid
  leaking information across stocks.

------------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------------------------

def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    # Download historical prices from Yahoo Finance and cache to data/prices.parquet.
    import yfinance as yf

    cache_path = DATA_DIR / "prices.parquet"
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.dropna(axis=1, how="all")  # drop tickers with no data at all
    prices.to_parquet(cache_path)
    return prices

# ------------------------------------------------------------------------------------

def load_cached_prices() -> pd.DataFrame:
    cache_path = DATA_DIR / "prices.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(
            "No cached prices found. Run data_pipeline.download_prices() first, "
            "or generate synthetic data with synthetic_prices() for local testing."
        )
    return pd.read_parquet(cache_path)

# ------------------------------------------------------------------------------------

def synthetic_prices(
    n_assets: int = 40, n_days: int = 1260, seed: int = 7
) -> pd.DataFrame:
    # Generate synthetic-but-realistic multi-asset price data for offline testing
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    tickers = [f"STK{i:02d}" for i in range(n_assets)]

    market = rng.normal(0.0004, 0.011, n_days)  # common market factor
    sector = rng.integers(0, 5, n_assets)  # 5 pseudo-sectors
    sector_factors = rng.normal(0.0, 0.006, (n_days, 5))

    betas = rng.uniform(0.5, 1.5, n_assets)
    idio_vol = rng.uniform(0.008, 0.025, n_assets)
    drift = rng.uniform(-0.0002, 0.0006, n_assets)

    returns = np.zeros((n_days, n_assets))
    for j in range(n_assets):
        idio = rng.normal(0, idio_vol[j], n_days)
        returns[:, j] = (
            drift[j] + betas[j] * market + sector_factors[:, sector[j]] + idio
        )

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)

# ------------------------------------------------------------------------------------

def engineer_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute per-ticker features from a wide price DataFrame...

    Returns a dict of wide DataFrames (date x ticker), one per feature, plus
    'returns' (simple daily returns) and 'fwd_return_21d' (the ML target: the
forward 21-day return, shifted so no lookahead).
    """
    returns = prices.pct_change()

    feats: dict[str, pd.DataFrame] = {}
    feats["returns"] = returns
    feats["mom_5d"] = prices.pct_change(5)
    feats["mom_21d"] = prices.pct_change(21)
    feats["vol_21d"] = returns.rolling(21).std()
    feats["vol_63d"] = returns.rolling(63).std()
    feats["ma_ratio_10_50"] = prices.rolling(10).mean() / prices.rolling(50).mean() - 1

    # RSI (14-day)
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, signal 9)
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feats["macd_hist"] = macd - signal

    # ML target: forward 21-day return (shifted -21 so the label at date t is known only using prices up to t+21 — this column must be dropped from any "features" matrix and used only as y).
    feats["fwd_return_21d"] = prices.pct_change(21).shift(-21)

    return feats

# ------------------------------------------------------------------------------------

def build_long_dataset(feats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    #Melt the dict-of-wide-frames into one long (date, ticker, feature...) table for feeding into a tabular ML model
    frames = []
    for name, wide in feats.items():
        long = wide.stack().rename(name)
        frames.append(long)
    df = pd.concat(frames, axis=1)
    df.index.set_names(["date", "ticker"], inplace=True)
    return df.reset_index()

# ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=str, default="config/nifty50.txt")
    ap.add_argument("--start", type=str, default="2019-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (no internet)")
    args = ap.parse_args()

    if args.synthetic:
        prices = synthetic_prices()
    else:
        with open(args.tickers) as f:
            tickers = [line.strip() for line in f if line.strip()]
        prices = download_prices(tickers, args.start, args.end)

    feats = engineer_features(prices)
    long_df = build_long_dataset(feats)
    long_df.to_parquet(DATA_DIR / "features_long.parquet")
    print(f"Saved {len(prices.columns)} tickers, {len(prices)} days -> data/features_long.parquet")

# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------------

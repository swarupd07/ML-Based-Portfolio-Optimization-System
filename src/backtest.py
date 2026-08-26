"""
Walk-forward backtest engine.

Rules enforced here:
  1. At each rebalance date t, every input (ML model training data, covariance
     matrix, expected returns) uses ONLY data strictly before t. No lookahead.
  2. Transaction costs are charged on every trade: cost = turnover * cost_bps.
  3. Turnover is capped per rebalance to avoid unrealistic "churn" in the portfolio.
  4. We run and compare FOUR strategies side by side:
       - Equal Weight            (the naive baseline)
       - Sample-Cov Markowitz    (Version of Markowitz using the sample covariance matrix)
       - Shrinkage-Cov Markowitz (our contribution: same optimizer, better Sigma)
       - Min-Variance (shrinkage)(a defensive anchor, ignores return forecasts)
"""
# ------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.covariance import sample_covariance, shrinkage_covariance
from src.ml_predict import train_predict, historical_mean_baseline, out_of_sample_ic
from src.optimizer import optimize_portfolio, min_variance_portfolio

# ------------------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    rebalance_freq: str = "ME"         # month-end
    cost_bps: float = 10.0             # 10 bps per unit turnover (one way)
    turnover_limit: float = 0.5        # max L1 turnover per rebalance
    max_weight: float = 0.15
    risk_aversion: float = 3.0
    lookback_min_days: int = 252       # need at least 1yr history

@dataclass
class StrategyResult:
    name: str
    weights_history: dict = field(default_factory=dict)   # date -> pd.Series
    returns: pd.Series = None                              # daily net-of-cost returns
    diagnostics: list = field(default_factory=list)

# ------------------------------------------------------------------------------------

def _rebalance_dates(dates: pd.DatetimeIndex, freq: str, min_start_idx: int) -> list:
    df = pd.Series(1, index=dates)
    period_starts = df.resample(freq).apply(lambda x: x.index.min()).dropna()
    valid = [d for d in period_starts if dates.get_loc(d) >= min_start_idx]
    return sorted(valid)

# ------------------------------------------------------------------------------------

def run_backtest(prices: pd.DataFrame, long_df: pd.DataFrame, config: BacktestConfig = None):
    # Runs all four strategies over the full price history and returns a dict of StrategyResult, plus a merged performance-metrics table.
    config = config or BacktestConfig()
    dates = prices.index
    daily_returns = prices.pct_change()

    rebal_dates = _rebalance_dates(dates, config.rebalance_freq, config.lookback_min_days)

    strategies = ["equal_weight", "sample_cov", "shrinkage_cov", "min_variance"]
    results = {s: StrategyResult(name=s) for s in strategies}
    ic_log = []

    prev_weights = {s: None for s in strategies}

    for i, rebal_date in enumerate(rebal_dates):
        window = daily_returns[daily_returns.index < rebal_date].tail(config.lookback_min_days)
        window = window.dropna(axis=1, how="any")
        if window.shape[1] < 5:
            continue
        tickers = window.columns

        # expected returns: ML prediction vs historical-mean baseline 
        ml_preds, diag = train_predict(long_df, train_end=rebal_date, predict_date=rebal_date)
        hist_mean = historical_mean_baseline(long_df, train_end=rebal_date)

        # blend: mostly historical mean (robust), a small ML tilt (since ML signal is weak — this reflects reality rather than pretending ML is reliable)
        mu = hist_mean.reindex(tickers).fillna(0)
        ml_tilt = ml_preds.reindex(tickers).fillna(0)
        expected_returns = mu + 5.0 * ml_tilt  # ML predicts 5-day return; scale to annual-ish tilt

        # Out-of-sample IC check (needs the actuals, only available once we
        # pass this rebalance date, so we log predicted vs what actually happened over the NEXT window for diagnostic purposes)
        future_actual = long_df[long_df["date"] == rebal_date].set_index("ticker")["fwd_return_21d"]
        ic_log.append({
            "date": rebal_date,
            "ic": out_of_sample_ic(ml_preds, future_actual),
            "n_train_rows": diag.get("n_train_rows"),
        })

        # covariance estimates 
        samp_cov = sample_covariance(window)
        shrink_cov, delta = shrinkage_covariance(window)

        # solving each strategy weights 
        w_eq = pd.Series(1 / len(tickers), index=tickers)

        w_samp = optimize_portfolio(
            expected_returns, samp_cov, risk_aversion=config.risk_aversion,
            max_weight=config.max_weight, prev_weights=prev_weights["sample_cov"],
            turnover_limit=config.turnover_limit,
        )

        w_shrink = optimize_portfolio(
            expected_returns, shrink_cov, risk_aversion=config.risk_aversion,
            max_weight=config.max_weight, prev_weights=prev_weights["shrinkage_cov"],
            turnover_limit=config.turnover_limit,
        )

        w_minvar = min_variance_portfolio(shrink_cov, max_weight=config.max_weight)

        weight_map = {
            "equal_weight": w_eq, "sample_cov": w_samp,
            "shrinkage_cov": w_shrink, "min_variance": w_minvar,
        }

        # Determining the holding period (until next rebalance date) 
        end_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        period_returns = daily_returns.loc[rebal_date:end_date, tickers].iloc[1:]

        for s in strategies:
            w = weight_map[s]
            w = w.reindex(tickers).fillna(0)
            w = w / w.sum() if w.sum() > 0 else w

            turnover = (w - prev_weights[s]).abs().sum() if prev_weights[s] is not None else w.abs().sum()
            cost = turnover * config.cost_bps / 10000

            gross_returns = period_returns @ w
            if len(gross_returns) > 0:
                gross_returns.iloc[0] -= cost  # charging cost on rebalance day
            results[s].weights_history[rebal_date] = w
            results[s].returns = pd.concat([results[s].returns, gross_returns]) if results[s].returns is not None else gross_returns

            prev_weights[s] = w

    ic_df = pd.DataFrame(ic_log)
    metrics = pd.DataFrame({s: performance_metrics(results[s].returns) for s in strategies}).T
    return results, metrics, ic_df

# ------------------------------------------------------------------------------------

def performance_metrics(returns: pd.Series, risk_free_rate: float = 0.065) -> dict:
    # Standard backtest metrics: total return, annualized vol, Sharpe, Sortino, max drawdown. Computed on realized (net-of-cost) daily returns
    returns = returns.dropna()
    if len(returns) == 0:
        return {k: np.nan for k in
                ["total_return", "annualized_return", "annualized_vol", "sharpe", "sortino", "max_drawdown"]}

    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
    sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol and downside_vol > 0 else np.nan

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()

    return {
        "total_return": total_return, "annualized_return": ann_return,
        "annualized_vol": ann_vol, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": max_dd,
    }

# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_pipeline import synthetic_prices, engineer_features, build_long_dataset

    prices = synthetic_prices(n_assets=25, n_days=756)  # 3yr synthetic, smaller for a fast test
    feats = engineer_features(prices)
    long_df = build_long_dataset(feats)

    cfg = BacktestConfig(lookback_min_days=126)  # shorter lookback as sample is small
    results, metrics, ic_df = run_backtest(prices, long_df, cfg)

    print("=== Performance ===")
    print(metrics.round(4))
    print("\n=== ML signal quality (mean out-of-sample IC) ===")
    print(f"mean IC: {ic_df['ic'].mean():.4f}  (values near 0 = no signal, which is expected/honest)")

# ------------------------------------------------------------------------------------

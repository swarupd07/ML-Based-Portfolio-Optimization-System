"""
Walk-forward backtest engine.

Rules enforced here
-------------------
1. ML training is purged so every 21-day forward target used for fitting is
   fully observable before the rebalance date.
2. Covariance / historical-return inputs use data before each rebalance.
3. Transaction costs are charged on every trade:
       cost = turnover * cost_bps
4. Turnover is capped per rebalance to avoid unrealistic portfolio churn.
5. Four baseline strategies are compared side by side:
       - Equal Weight
       - Sample-Cov Markowitz
       - Shrinkage-Cov Markowitz
       - Min-Variance (shrinkage)

The Black-Litterman extension is evaluated separately in
run_black_litterman_backtest.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.covariance import sample_covariance, shrinkage_covariance
from src.ml_predict import (
    train_predict,
    historical_mean_baseline,
    out_of_sample_ic,
)
from src.optimizer import (
    optimize_portfolio,
    min_variance_portfolio,
)


# ------------------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    rebalance_freq: str = "ME"
    cost_bps: float = 10.0
    turnover_limit: float = 0.5
    max_weight: float = 0.15
    risk_aversion: float = 3.0
    lookback_min_days: int = 252


@dataclass
class StrategyResult:
    name: str
    weights_history: dict = field(default_factory=dict)
    returns: pd.Series | None = None
    diagnostics: list = field(default_factory=list)


# ------------------------------------------------------------------------------------
def _rebalance_dates(
    dates: pd.DatetimeIndex,
    freq: str,
    min_start_idx: int,
) -> list[pd.Timestamp]:
    """
    Use the first trading date in each monthly bucket as the rebalance date.

    Returns from the rebalance date itself are excluded from the subsequent
    holding-period return calculation, so the portfolio is effectively formed
    after observing that date's features and held from the next session onward.
    """
    df = pd.Series(1, index=dates)

    period_starts = (
        df.resample(freq)
        .apply(lambda x: x.index.min())
        .dropna()
    )

    valid = [
        d
        for d in period_starts
        if dates.get_loc(d) >= min_start_idx
    ]

    return sorted(valid)


# ------------------------------------------------------------------------------------
def run_backtest(
    prices: pd.DataFrame,
    long_df: pd.DataFrame,
    config: BacktestConfig | None = None,
):
    """
    Run the four original baseline strategies over the full price history.
    """
    config = config or BacktestConfig()

    dates = prices.index
    daily_returns = prices.pct_change()

    rebal_dates = _rebalance_dates(
        dates=dates,
        freq=config.rebalance_freq,
        min_start_idx=config.lookback_min_days,
    )

    strategies = [
        "equal_weight",
        "sample_cov",
        "shrinkage_cov",
        "min_variance",
    ]

    results = {
        s: StrategyResult(name=s)
        for s in strategies
    }

    ic_log = []

    prev_weights = {
        s: None
        for s in strategies
    }

    for i, rebal_date in enumerate(rebal_dates):

        window = (
            daily_returns[
                daily_returns.index < rebal_date
            ]
            .tail(config.lookback_min_days)
            .dropna(axis=1, how="any")
        )

        if window.shape[1] < 5:
            continue

        tickers = window.columns

        # ------------------------------------------------------------------
        # Expected returns
        # ------------------------------------------------------------------
        # train_predict() now purges the final 21 trading dates from training
        # so no forward-return target crosses the rebalance boundary.
        ml_preds, diag = train_predict(
            long_df=long_df,
            train_end=rebal_date,
            predict_date=rebal_date,
        )

        hist_mean = historical_mean_baseline(
            long_df=long_df,
            train_end=rebal_date,
        )

        mu = hist_mean.reindex(tickers).fillna(0.0)
        ml_tilt = ml_preds.reindex(tickers).fillna(0.0)

        # Historical baseline retained exactly as an experiment:
        # 5.0 is a heuristic ML tilt coefficient, NOT a 21-day annualization.
        ML_TILT_SCALE = 5.0

        expected_returns = (
            mu
            + ML_TILT_SCALE * ml_tilt
        )

        # ------------------------------------------------------------------
        # OOS IC diagnostic
        # ------------------------------------------------------------------
        # This uses future realized return ONLY for evaluation. It does not
        # enter model fitting or portfolio construction.
        future_actual = (
            long_df[
                long_df["date"] == rebal_date
            ]
            .set_index("ticker")["fwd_return_21d"]
        )

        ic_log.append(
            {
                "date": rebal_date,
                "ic": out_of_sample_ic(
                    ml_preds,
                    future_actual,
                ),
                "n_train_rows": diag.get("n_train_rows"),
                "purged_through": diag.get("purged_through"),
            }
        )

        # ------------------------------------------------------------------
        # Covariance estimates
        # ------------------------------------------------------------------
        samp_cov = sample_covariance(window)
        shrink_cov, delta = shrinkage_covariance(window)

        # ------------------------------------------------------------------
        # Portfolio weights
        # ------------------------------------------------------------------
        w_eq = pd.Series(
            1.0 / len(tickers),
            index=tickers,
        )

        w_samp = optimize_portfolio(
            expected_returns,
            samp_cov,
            risk_aversion=config.risk_aversion,
            max_weight=config.max_weight,
            prev_weights=prev_weights["sample_cov"],
            turnover_limit=config.turnover_limit,
        )

        w_shrink = optimize_portfolio(
            expected_returns,
            shrink_cov,
            risk_aversion=config.risk_aversion,
            max_weight=config.max_weight,
            prev_weights=prev_weights["shrinkage_cov"],
            turnover_limit=config.turnover_limit,
        )

        w_minvar = min_variance_portfolio(
            shrink_cov,
            max_weight=config.max_weight,
        )

        weight_map = {
            "equal_weight": w_eq,
            "sample_cov": w_samp,
            "shrinkage_cov": w_shrink,
            "min_variance": w_minvar,
        }

        # ------------------------------------------------------------------
        # Holding period
        # ------------------------------------------------------------------
        end_date = (
            rebal_dates[i + 1]
            if i + 1 < len(rebal_dates)
            else dates[-1]
        )

        period_returns = daily_returns.loc[
            rebal_date:end_date,
            tickers,
        ].iloc[1:]

        for strategy in strategies:

            w = (
                weight_map[strategy]
                .reindex(tickers)
                .fillna(0.0)
            )

            if w.sum() > 0:
                w = w / w.sum()

            if prev_weights[strategy] is None:
                turnover = w.abs().sum()
            else:
                turnover = (
                    w
                    - prev_weights[strategy]
                    .reindex(tickers)
                    .fillna(0.0)
                ).abs().sum()

            cost = (
                turnover
                * config.cost_bps
                / 10000
            )

            gross_returns = period_returns @ w

            if len(gross_returns) > 0:
                gross_returns.iloc[0] -= cost

            results[strategy].weights_history[
                rebal_date
            ] = w

            if results[strategy].returns is None:
                results[strategy].returns = gross_returns
            else:
                results[strategy].returns = pd.concat(
                    [
                        results[strategy].returns,
                        gross_returns,
                    ]
                )

            prev_weights[strategy] = w

    ic_df = pd.DataFrame(ic_log)

    metrics = pd.DataFrame(
        {
            s: performance_metrics(
                results[s].returns
            )
            for s in strategies
        }
    ).T

    return results, metrics, ic_df


# ------------------------------------------------------------------------------------
def performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.065,
) -> dict:
    """
    Standard realized net-of-cost portfolio metrics.
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return {
            k: np.nan
            for k in [
                "total_return",
                "annualized_return",
                "annualized_vol",
                "sharpe",
                "sortino",
                "max_drawdown",
            ]
        }

    total_return = (
        (1 + returns).prod()
        - 1
    )

    n_years = len(returns) / 252

    ann_return = (
        (1 + total_return) ** (1 / n_years)
        - 1
        if n_years > 0
        else np.nan
    )

    ann_vol = (
        returns.std()
        * np.sqrt(252)
    )

    sharpe = (
        (ann_return - risk_free_rate)
        / ann_vol
        if ann_vol > 0
        else np.nan
    )

    downside = returns[
        returns < 0
    ]

    downside_vol = (
        downside.std() * np.sqrt(252)
        if len(downside) > 0
        else np.nan
    )

    sortino = (
        (ann_return - risk_free_rate)
        / downside_vol
        if pd.notna(downside_vol)
        and downside_vol > 0
        else np.nan
    )

    cumulative = (
        1 + returns
    ).cumprod()

    running_max = cumulative.cummax()

    drawdown = (
        cumulative
        / running_max
        - 1
    )

    max_dd = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
    }


# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_pipeline import (
        synthetic_prices,
        engineer_features,
        build_long_dataset,
    )

    prices = synthetic_prices(
        n_assets=25,
        n_days=756,
    )

    feats = engineer_features(prices)
    long_df = build_long_dataset(feats)

    cfg = BacktestConfig(
        lookback_min_days=126
    )

    results, metrics, ic_df = run_backtest(
        prices,
        long_df,
        cfg,
    )

    print("=== Performance ===")
    print(metrics.round(4))

    print("\n=== ML signal quality ===")
    print(
        "mean OOS IC:",
        round(ic_df["ic"].mean(), 4),
    )

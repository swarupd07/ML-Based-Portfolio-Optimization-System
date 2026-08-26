"""
Black-Litterman + ML walk-forward experiment.

This script compares:

1. BL Prior Only
2. BL + ML relative views

Important validity rule
-----------------------
src.ml_predict.train_predict() purges the final 21 trading dates from every
training sample, ensuring no 21-day forward-return label crosses the rebalance
boundary.

Default final research configuration
------------------------------------
- BL market risk aversion δ = 2.5
- tau = 0.05
- relative views = 10
- uncertainty multiplier = 3
- BL covariance window = up to 756 trading days
- portfolio-risk covariance window = 252 trading days
- portfolio risk aversion = 3
- max stock weight = 15%
- turnover limit = 0.5
- transaction cost = 10 bps per unit turnover
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.data_pipeline import (
    engineer_features,
    build_long_dataset,
)
from src.covariance import shrinkage_covariance
from src.ml_predict import train_predict
from src.optimizer import optimize_portfolio
from src.black_litterman import (
    build_bl_expected_returns,
    implied_equilibrium_returns,
)
from src.market_data import (
    align_shares_to_prices,
    build_market_caps,
    get_market_weights_asof,
)
from src.backtest import performance_metrics


# ------------------------------------------------------------------------------------
DEFAULT_UNCERTAINTY_MULTIPLIER = 3.0

MARKET_RISK_AVERSION = 2.5
TAU = 0.05
N_VIEWS = 10

PORTFOLIO_RISK_AVERSION = 3.0
MAX_WEIGHT = 0.15
TURNOVER_LIMIT = 0.5
COST_BPS = 10.0

RISK_WINDOW_DAYS = 252
BL_WINDOW_DAYS = 756


# ------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BL prior-only vs BL+ML walk-forward backtest."
    )

    parser.add_argument(
        "--uncertainty-multiplier",
        type=float,
        default=DEFAULT_UNCERTAINTY_MULTIPLIER,
        help=(
            "Multiplier applied to Omega. "
            "Higher = less confidence in ML views. "
            "Default: 3."
        ),
    )

    return parser.parse_args()


# ------------------------------------------------------------------------------------
def calculate_turnover(
    weights: pd.Series,
    prev_weights: pd.Series | None,
) -> float:
    """
    L1 portfolio turnover relative to the previous rebalance.
    """
    if prev_weights is None:
        return float(
            weights.abs().sum()
        )

    aligned_prev = (
        prev_weights
        .reindex(weights.index)
        .fillna(0.0)
    )

    return float(
        (
            weights
            - aligned_prev
        ).abs().sum()
    )


# ------------------------------------------------------------------------------------
def apply_transaction_cost(
    realized_returns: pd.Series,
    turnover: float,
    cost_bps: float = COST_BPS,
) -> pd.Series:
    """
    Charge transaction cost on the first holding-period return observation.
    """
    realized_returns = realized_returns.copy()

    if len(realized_returns) > 0:
        cost = (
            turnover
            * cost_bps
            / 10000
        )

        realized_returns.iloc[0] -= cost

    return realized_returns


# ------------------------------------------------------------------------------------
def main(
    uncertainty_multiplier: float,
):
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    prices = pd.read_parquet(
        "data/prices.parquet"
    )

    shares_df = pd.read_csv(
        "data/shares_outstanding.csv"
    )

    feats = engineer_features(prices)
    long_df = build_long_dataset(feats)

    aligned_shares = align_shares_to_prices(
        shares_df,
        prices,
    )

    market_caps = build_market_caps(
        aligned_shares,
        prices,
    )

    daily_returns = prices.pct_change()

    # ------------------------------------------------------------------
    # Rebalance dates
    # ------------------------------------------------------------------
    rebalance_dates = (
        pd.Series(
            1,
            index=prices.index,
        )
        .resample("ME")
        .apply(
            lambda x: x.index.min()
        )
        .dropna()
    )

    rebalance_dates = [
        d
        for d in rebalance_dates
        if prices.index.get_loc(d)
        >= RISK_WINDOW_DAYS
    ]

    # ------------------------------------------------------------------
    # Histories
    # ------------------------------------------------------------------
    prev_bl_weights = None
    prev_prior_weights = None

    bl_portfolio_returns = []
    prior_portfolio_returns = []

    diagnostics = []

    # ------------------------------------------------------------------
    # Walk-forward loop
    # ------------------------------------------------------------------
    for current_idx, rebal_date in enumerate(
        rebalance_dates
    ):

        # ----------------------------------------------
        # Current portfolio-risk covariance
        # ----------------------------------------------
        risk_window = (
            daily_returns[
                daily_returns.index
                < rebal_date
            ]
            .tail(RISK_WINDOW_DAYS)
            .dropna(
                axis=1,
                how="any",
            )
        )

        # ----------------------------------------------
        # Longer BL equilibrium covariance
        # ----------------------------------------------
        bl_window = (
            daily_returns[
                daily_returns.index
                < rebal_date
            ]
            .tail(BL_WINDOW_DAYS)
        )

        if len(bl_window) == 0:
            continue

        min_obs = int(
            0.95 * len(bl_window)
        )

        bl_window = bl_window.dropna(
            axis=1,
            thresh=min_obs,
        )

        bl_window = (
            bl_window
            .ffill()
            .dropna()
        )

        if risk_window.shape[1] < 5:
            continue

        common_tickers = (
            risk_window.columns
            .intersection(
                bl_window.columns
            )
        )

        if len(common_tickers) < 5:
            continue

        risk_window = risk_window[
            common_tickers
        ]

        bl_window = bl_window[
            common_tickers
        ]

        tickers = common_tickers

        # ----------------------------------------------
        # Covariance estimates
        # ----------------------------------------------
        shrink_cov, _ = (
            shrinkage_covariance(
                risk_window
            )
        )

        bl_cov, _ = (
            shrinkage_covariance(
                bl_window
            )
        )

        # ----------------------------------------------
        # Point-in-time market weights
        # ----------------------------------------------
        market_weights = (
            get_market_weights_asof(
                market_caps=market_caps,
                rebalance_date=rebal_date,
                tickers=tickers,
            )
        )

        # ----------------------------------------------
        # Leakage-safe 21-day ML predictions
        # ----------------------------------------------
        ml_preds, ml_diag = train_predict(
            long_df=long_df,
            train_end=rebal_date,
            predict_date=rebal_date,
        )

        # ----------------------------------------------
        # BL equilibrium prior
        # ----------------------------------------------
        prior = (
            implied_equilibrium_returns(
                cov_matrix=bl_cov,
                market_weights=market_weights,
                risk_aversion=MARKET_RISK_AVERSION,
            )
        )

        # ----------------------------------------------
        # Prior-only portfolio
        # ----------------------------------------------
        prior_weights = optimize_portfolio(
            expected_returns=prior,
            cov_matrix=shrink_cov,
            risk_aversion=PORTFOLIO_RISK_AVERSION,
            max_weight=MAX_WEIGHT,
            prev_weights=prev_prior_weights,
            turnover_limit=TURNOVER_LIMIT,
        )

        # ----------------------------------------------
        # BL + ML posterior
        # ----------------------------------------------
        mu_bl = build_bl_expected_returns(
            cov_matrix=bl_cov,
            market_weights=market_weights,
            risk_aversion=MARKET_RISK_AVERSION,
            ml_predictions=ml_preds,
            tau=TAU,
            n_views=N_VIEWS,
            uncertainty_multiplier=uncertainty_multiplier,
        )

        bl_weights = optimize_portfolio(
            expected_returns=mu_bl,
            cov_matrix=shrink_cov,
            risk_aversion=PORTFOLIO_RISK_AVERSION,
            max_weight=MAX_WEIGHT,
            prev_weights=prev_bl_weights,
            turnover_limit=TURNOVER_LIMIT,
        )

        # ----------------------------------------------
        # Holding period
        # ----------------------------------------------
        if current_idx + 1 < len(
            rebalance_dates
        ):
            next_date = (
                rebalance_dates[
                    current_idx + 1
                ]
            )
        else:
            next_date = (
                prices.index[-1]
            )

        period_returns = (
            daily_returns.loc[
                rebal_date:next_date,
                tickers,
            ]
            .iloc[1:]
        )

        # ----------------------------------------------
        # Prior-only realized returns
        # ----------------------------------------------
        prior_turnover = (
            calculate_turnover(
                prior_weights,
                prev_prior_weights,
            )
        )

        prior_realized = (
            period_returns
            @ prior_weights
        )

        prior_realized = (
            apply_transaction_cost(
                prior_realized,
                prior_turnover,
            )
        )

        prior_portfolio_returns.append(
            prior_realized
        )

        # ----------------------------------------------
        # BL + ML realized returns
        # ----------------------------------------------
        bl_turnover = (
            calculate_turnover(
                bl_weights,
                prev_bl_weights,
            )
        )

        bl_realized = (
            period_returns
            @ bl_weights
        )

        bl_realized = (
            apply_transaction_cost(
                bl_realized,
                bl_turnover,
            )
        )

        bl_portfolio_returns.append(
            bl_realized
        )

        diagnostics.append(
            {
                "date": rebal_date,
                "n_assets": len(tickers),
                "n_train_rows": ml_diag.get(
                    "n_train_rows"
                ),
                "purged_through": ml_diag.get(
                    "purged_through"
                ),
                "prior_mean": prior.mean(),
                "posterior_mean": mu_bl.mean(),
                "bl_turnover": bl_turnover,
                "prior_turnover": prior_turnover,
            }
        )

        prev_bl_weights = bl_weights
        prev_prior_weights = (
            prior_weights
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    if not bl_portfolio_returns:
        raise RuntimeError(
            "No BL portfolio returns were generated."
        )

    if not prior_portfolio_returns:
        raise RuntimeError(
            "No prior-only portfolio returns were generated."
        )

    bl_returns = (
        pd.concat(
            bl_portfolio_returns
        )
        .sort_index()
    )

    prior_returns = (
        pd.concat(
            prior_portfolio_returns
        )
        .sort_index()
    )

    bl_metrics = (
        performance_metrics(
            bl_returns
        )
    )

    prior_metrics = (
        performance_metrics(
            prior_returns
        )
    )

    comparison = pd.DataFrame(
        {
            "BL + ML": bl_metrics,
            "BL Prior Only": prior_metrics,
        }
    ).T

    print(
        "\n=== Leakage-safe BL comparison "
        f"(uncertainty_multiplier={uncertainty_multiplier:g}) ==="
    )

    print(
        comparison.round(4)
    )

    # Helpful validity diagnostic.
    diagnostics_df = pd.DataFrame(
        diagnostics
    )

    if not diagnostics_df.empty:
        print(
            "\n=== Purge diagnostic ==="
        )

        print(
            diagnostics_df[
                [
                    "date",
                    "purged_through",
                    "n_train_rows",
                ]
            ]
            .head(8)
            .to_string(index=False)
        )

    return (
        comparison,
        bl_returns,
        prior_returns,
        diagnostics_df,
    )


# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    main(
        uncertainty_multiplier=(
            args.uncertainty_multiplier
        )
    )

import pandas as pd

from src.data_pipeline import engineer_features, build_long_dataset
from src.covariance import shrinkage_covariance
from src.ml_predict import train_predict
from src.optimizer import optimize_portfolio
from src.black_litterman import build_bl_expected_returns, implied_equilibrium_returns
from src.market_data import (
    align_shares_to_prices,
    build_market_caps,
    get_market_weights_asof,
)
from src.backtest import performance_metrics

UNCERTAINTY_MULTIPLIER = 15

# =====================================================================================================
prices = pd.read_parquet("data/prices.parquet")

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

# =====================================================================================================

daily_returns = prices.pct_change()

rebalance_dates = (
    pd.Series(1, index=prices.index)
    .resample("ME")
    .apply(lambda x: x.index.min())
    .dropna()
)

rebalance_dates = [
    d for d in rebalance_dates
    if prices.index.get_loc(d) >= 252
]

# =====================================================================================================

bl_returns_history = {}

prev_weights = None
bl_weights_history = {}
bl_portfolio_returns = []
prev_prior_weights = None
prior_portfolio_returns = []

for current_idx, rebal_date in enumerate(rebalance_dates):

    window = (
        daily_returns[
            daily_returns.index < rebal_date
        ]
        .tail(252)
        .dropna(axis=1, how="any")
    )

    bl_window = (
    daily_returns[
        daily_returns.index < rebal_date
    ]
    .tail(756)
)

    # keep stocks with enough usable observations
    min_obs = int(0.95 * len(bl_window))

    bl_window = bl_window.dropna(
        axis=1,
        thresh=min_obs,
    )

    # fill small remaining gaps safely
    bl_window = bl_window.ffill().dropna()



    if window.shape[1] < 5:
        continue


    common_tickers = window.columns.intersection(bl_window.columns)

    window = window[common_tickers]
    bl_window = bl_window[common_tickers]

    tickers = common_tickers

    # Ledoit-Wolf covariance
    shrink_cov, _ = shrinkage_covariance(window)
    bl_cov, _ = shrinkage_covariance(bl_window)

    # Point-in-time market weights
    w_mkt = get_market_weights_asof(
        market_caps=market_caps,
        rebalance_date=rebal_date,
        tickers=tickers,
    )

    # ML 21-day predictions
    ml_preds, _ = train_predict(
        long_df=long_df,
        train_end=rebal_date,
        predict_date=rebal_date,
    )

    prior = implied_equilibrium_returns(
    cov_matrix=bl_cov,
    market_weights=w_mkt,
    risk_aversion=2.5,)

    prior_weights = optimize_portfolio(
    expected_returns=prior,
    cov_matrix=shrink_cov,
    risk_aversion=3.0,
    max_weight=0.15,
    prev_weights=prev_prior_weights,
    turnover_limit=0.5,
)
    
    # Black-Litterman posterior expected returns
    mu_bl = build_bl_expected_returns(
        cov_matrix=bl_cov,
        market_weights=w_mkt,
        risk_aversion=2.5,
        ml_predictions=ml_preds,
        tau=0.05,
        n_views=10,
        uncertainty_multiplier=UNCERTAINTY_MULTIPLIER,
    )

    weights = optimize_portfolio(
    expected_returns=mu_bl,
    cov_matrix=shrink_cov,
    risk_aversion=3.0,
    max_weight=0.15,
    prev_weights=prev_weights,
    turnover_limit=0.5,
)

    if current_idx + 1 < len(rebalance_dates):
        next_date = rebalance_dates[current_idx + 1]
    else:
        next_date = prices.index[-1]

    period_returns = daily_returns.loc[
        rebal_date:next_date,
        tickers
    ].iloc[1:]


    bl_weights_history[rebal_date] = weights

    if prev_weights is None:
        turnover = weights.abs().sum()
    else:
        turnover = (
            weights
            - prev_weights.reindex(weights.index).fillna(0.0)
        ).abs().sum()

    cost = turnover * 10.0 / 10000


    if prev_prior_weights is None:
        prior_turnover = prior_weights.abs().sum()
    else:
        prior_turnover = (
            prior_weights
            - prev_prior_weights.reindex(prior_weights.index).fillna(0.0)
        ).abs().sum()

    prior_cost = prior_turnover * 10.0 / 10000

    prior_realized = period_returns @ prior_weights

    if len(prior_realized) > 0:
        prior_realized.iloc[0] -= prior_cost

    prior_portfolio_returns.append(prior_realized)

    prev_prior_weights = prior_weights

    realized = period_returns @ weights

    if len(realized) > 0:
        realized.iloc[0] -= cost

    bl_portfolio_returns.append(realized)

    prev_weights = weights

    '''print(
    rebal_date.date(),
    "prior:",
    round(prior.mean(), 4),
    "posterior:",
    round(mu_bl.mean(), 4),
    "max:",
    round(mu_bl.max(), 4))

    print(
    rebal_date.date(),
    "weight sum:",
    round(weights.sum(), 4),
    "max weight:",
    round(weights.max(), 4),
    "positions:",
    int((weights > 0.001).sum()))'''


    bl_returns_history[rebal_date] = mu_bl

    ''' print(
        rebal_date.date(),
        "assets:", len(mu_bl),
        "mean BL return:",
        round(mu_bl.mean(), 4),
    )'''


bl_returns = pd.concat(bl_portfolio_returns).sort_index()

bl_metrics = performance_metrics(bl_returns)

'''print("\n=== Black-Litterman Performance ===")

for key, value in bl_metrics.items():
    print(key, round(value, 4))
'''

prior_returns = pd.concat(prior_portfolio_returns).sort_index()

prior_metrics = performance_metrics(prior_returns)

'''print("\n=== BL Prior-Only Performance ===")

for key, value in prior_metrics.items():
    print(key, round(value, 4))
'''

comparison = pd.DataFrame({
    "BL + ML": bl_metrics,
    "BL Prior Only": prior_metrics,
}).T

print(f"\n=== BL Comparison with uncertainty_multiplier= {UNCERTAINTY_MULTIPLIER } ===")
print(comparison.round(4))
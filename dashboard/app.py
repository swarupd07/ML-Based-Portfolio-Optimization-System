# Streamlit UI for the leakage-safe 21-day + Black-Litterman portfolio system

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_pipeline import (
    load_cached_prices,
    engineer_features,
    build_long_dataset,
)
from src.covariance import shrinkage_covariance
from src.optimizer import optimize_portfolio, portfolio_stats
from src.ml_predict import train_predict
from src.black_litterman import (
    build_bl_expected_returns,
    implied_equilibrium_returns,
)
from src.market_data import (
    align_shares_to_prices,
    build_market_caps,
    get_market_weights_asof,
)


# ======================================================================================
# Page
# ======================================================================================

st.set_page_config(
    page_title="Black-Litterman Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

st.title("Black-Litterman Portfolio Optimizer")

st.caption(
    "Leakage-safe 21-day ML views + market-implied Black-Litterman expected returns + "
    "Ledoit-Wolf covariance estimation + constrained Markowitz allocation."
)

ROOT = Path(__file__).resolve().parent.parent
PRICES_PATH = ROOT / "data" / "prices.parquet"
SHARES_PATH = ROOT / "data" / "shares_outstanding.csv"


# ======================================================================================
# Final research configuration
# ======================================================================================

# This is intentionally fixed to the best observed leakage-safe confidence setting from the completed sweep.
PORTFOLIO_RISK_AVERSION = 3.0
MAX_WEIGHT = 0.15

MARKET_RISK_AVERSION = 2.5
TAU = 0.05
N_VIEWS = 10
UNCERTAINTY_MULTIPLIER = 1.0

RISK_WINDOW_DAYS = 252
BL_WINDOW_DAYS = 756

# Backtest settings
TURNOVER_LIMIT = 0.5
TRANSACTION_COST_BPS = 10.0

# Leakage-safe completed walk-forward results for the final displayed configuration.
FINAL_METRICS = {
    "annualized_return": 0.1712,
    "annualized_vol": 0.1618,
    "sharpe": 0.6563,
    "sortino": 0.7927,
    "max_drawdown": -0.2783,
    "total_return": 1.7635,
}

ML_OOS_IC = 0.022487277592645443


# ======================================================================================
# Sidebar
# ======================================================================================

st.sidebar.header("Portfolio")

investment_amount = st.sidebar.number_input(
    "Investment amount (₹)",
    min_value=10_000,
    value=1_000_000,
    step=10_000,
)

st.sidebar.caption(
    "The allocation uses the final selected leakage-safe BL + ML configuration."
)

st.sidebar.divider()

st.sidebar.header("Final strategy")

st.sidebar.markdown(
    f"""
**ML horizon:** 21 trading days  
**Portfolio risk aversion:** {PORTFOLIO_RISK_AVERSION:g}  
**Maximum stock weight:** {MAX_WEIGHT:.0%}  
**Market risk aversion δ:** {MARKET_RISK_AVERSION:g}  
**τ:** {TAU:g}  
**Relative ML views:** {N_VIEWS}  
**View uncertainty multiplier:** {UNCERTAINTY_MULTIPLIER:g}  
**Portfolio-risk window:** {RISK_WINDOW_DAYS} days  
**BL covariance window:** {BL_WINDOW_DAYS} days
"""
)


# ======================================================================================
# Data
# ======================================================================================

@st.cache_data(show_spinner=False)
def load_project_data():
    if not PRICES_PATH.exists():
        raise FileNotFoundError(
            "Missing data/prices.parquet. Run the data pipeline first."
        )

    prices = (
        load_cached_prices()
        .dropna(axis=1, how="all")
    )

    features = engineer_features(prices)
    long_df = build_long_dataset(features)

    return prices, long_df


@st.cache_data(show_spinner=False)
def load_market_caps(prices: pd.DataFrame):
    if not SHARES_PATH.exists():
        raise FileNotFoundError(
            "Missing data/shares_outstanding.csv. "
            "Run python download_market_data.py first."
        )

    shares_df = pd.read_csv(SHARES_PATH)

    aligned_shares = align_shares_to_prices(
        shares_df,
        prices,
    )

    return build_market_caps(
        aligned_shares,
        prices,
    )


try:
    prices, long_df = load_project_data()
    market_caps = load_market_caps(prices)

except Exception as exc:
    st.error(str(exc))
    st.stop()


daily_returns = prices.pct_change()

latest_price_date = pd.Timestamp(prices.index[-1])
latest_feature_date = pd.to_datetime(long_df["date"]).max()

st.caption(
    f"Data: **{len(prices.columns)} NSE large-cap stocks** | "
    f"{prices.index[0].date()} → {latest_price_date.date()} | "
    f"latest usable feature date: **{latest_feature_date.date()}**"
)


# ======================================================================================
# Current leakage-safe model state
# ======================================================================================

risk_window = (
    daily_returns[
        daily_returns.index < latest_price_date
    ]
    .tail(RISK_WINDOW_DAYS)
    .dropna(axis=1, how="any")
)

bl_window = (
    daily_returns[
        daily_returns.index < latest_price_date
    ]
    .tail(BL_WINDOW_DAYS)
)

min_obs = int(
    0.95 * len(bl_window)
)

bl_window = (
    bl_window
    .dropna(
        axis=1,
        thresh=min_obs,
    )
    .ffill()
    .dropna()
)

tickers = (
    risk_window.columns
    .intersection(
        bl_window.columns
    )
)

if len(tickers) < 5:
    st.error(
        "Too few assets have sufficient data for the current estimation windows."
    )
    st.stop()

risk_window = risk_window[tickers]
bl_window = bl_window[tickers]

risk_cov, risk_shrinkage = shrinkage_covariance(
    risk_window
)

bl_cov, bl_shrinkage = shrinkage_covariance(
    bl_window
)

market_weights = get_market_weights_asof(
    market_caps=market_caps,
    rebalance_date=latest_price_date,
    tickers=tickers,
)

# Leakage-safe because the updated train_predict() purges the final
# 21 trading dates from every training sample.
ml_predictions, ml_diag = train_predict(
    long_df=long_df,
    train_end=latest_price_date,
    predict_date=latest_feature_date,
)

prior = implied_equilibrium_returns(
    cov_matrix=bl_cov,
    market_weights=market_weights,
    risk_aversion=MARKET_RISK_AVERSION,
)

bl_expected_returns = build_bl_expected_returns(
    cov_matrix=bl_cov,
    market_weights=market_weights,
    risk_aversion=MARKET_RISK_AVERSION,
    ml_predictions=ml_predictions,
    tau=TAU,
    n_views=N_VIEWS,
    uncertainty_multiplier=UNCERTAINTY_MULTIPLIER,
)

# Latest allocation snapshot produced by the final model.
#
# Historical backtest performance shown at the top includes the turnover
# constraint. This latest snapshot is generated from the latest posterior
# without reconstructing the entire historical weight path.
weights = optimize_portfolio(
    expected_returns=bl_expected_returns,
    cov_matrix=risk_cov,
    risk_aversion=PORTFOLIO_RISK_AVERSION,
    max_weight=MAX_WEIGHT,
)

current_stats = portfolio_stats(
    weights,
    bl_expected_returns,
    risk_cov,
)


# ======================================================================================
# 1. Final portfolio
# ======================================================================================

st.header("1. Final BL + ML Portfolio")

st.success(
    "Portfolio generated with the final leakage-safe 21-day Black-Litterman + ML methodology."
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Walk-forward Sharpe",
    f"{FINAL_METRICS['sharpe']:.4f}",
)

m2.metric(
    "Walk-forward annual return",
    f"{FINAL_METRICS['annualized_return']:.2%}",
)

m3.metric(
    "Walk-forward volatility",
    f"{FINAL_METRICS['annualized_vol']:.2%}",
)

m4.metric(
    "Walk-forward max drawdown",
    f"{FINAL_METRICS['max_drawdown']:.2%}",
)

st.caption(
    "The metrics above are realized out-of-sample results for the selected "
    "uncertainty-multiplier=1 configuration. The weights below are the latest "
    "allocation generated by the same return/risk model from current data."
)


# ======================================================================================
# Allocation table
# ======================================================================================

allocation = (
    weights[
        weights > 0.001
    ]
    .sort_values(
        ascending=False
    )
)

allocation_df = pd.DataFrame(
    {
        "Ticker": allocation.index,
        "Weight": allocation.values,
        "Weight %": allocation.values * 100,
        "Amount (₹)": allocation.values * investment_amount,
        "BL Expected Return": (
            bl_expected_returns
            .reindex(allocation.index)
            .values
        ),
    }
)

display_allocation = allocation_df.copy()

display_allocation["Weight %"] = (
    display_allocation["Weight %"]
    .map(
        lambda x: f"{x:.2f}%"
    )
)

display_allocation["Amount (₹)"] = (
    display_allocation["Amount (₹)"]
    .map(
        lambda x: f"₹{x:,.0f}"
    )
)

display_allocation["BL Expected Return"] = (
    display_allocation["BL Expected Return"]
    * 100
).map(
    lambda x: f"{x:.2f}%"
)

display_allocation = display_allocation.drop(
    columns=["Weight"]
)

left, right = st.columns(
    [1, 1]
)

with left:
    fig_allocation = px.pie(
        allocation_df,
        names="Ticker",
        values="Weight",
        title="Latest BL + ML portfolio allocation",
    )

    st.plotly_chart(
        fig_allocation,
        use_container_width=True,
    )

with right:
    st.dataframe(
        display_allocation,
        use_container_width=True,
        hide_index=True,
        height=500,
    )


with st.expander(
    "Current model-implied snapshot",
    expanded=False,
):
    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Current expected return",
        f"{current_stats['expected_return']:.2%}",
    )

    s2.metric(
        "Current expected volatility",
        f"{current_stats['volatility']:.2%}",
    )

    s3.metric(
        "Current ex-ante Sharpe",
        f"{current_stats['sharpe_ratio']:.2f}",
    )

    s4.metric(
        "Active positions",
        int(
            (
                weights > 0.001
            ).sum()
        ),
    )

    st.caption(
        "These are current model-implied quantities and are not the historical "
        "walk-forward performance metrics shown above."
    )


# ======================================================================================
# 2. Current BL state
# ======================================================================================

st.header("2. Current Black-Litterman Model State")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Mean BL prior",
    f"{prior.mean():.2%}",
)

c2.metric(
    "Mean BL posterior",
    f"{bl_expected_returns.mean():.2%}",
)

c3.metric(
    "Largest BL posterior",
    f"{bl_expected_returns.max():.2%}",
)

c4.metric(
    "Leakage-safe OOS IC",
    f"{ML_OOS_IC:.4f}",
    help=(
        "Mean walk-forward Spearman IC from the corrected 21-day model. "
        "Training samples purge the last 21 trading dates before each rebalance."
    ),
)

state_df = pd.DataFrame(
    {
        "Ticker": tickers,
        "Market Weight": (
            market_weights
            .reindex(tickers)
            .values
        ),
        "ML 21d Prediction": (
            ml_predictions
            .reindex(tickers)
            .values
        ),
        "BL Prior": (
            prior
            .reindex(tickers)
            .values
        ),
        "BL Posterior": (
            bl_expected_returns
            .reindex(tickers)
            .values
        ),
        "Portfolio Weight": (
            weights
            .reindex(tickers)
            .values
        ),
    }
)

state_display = state_df.copy()

for col in [
    "Market Weight",
    "ML 21d Prediction",
    "BL Prior",
    "BL Posterior",
    "Portfolio Weight",
]:
    state_display[col] = (
        state_display[col]
        * 100
    ).map(
        lambda x: f"{x:.2f}%"
    )

st.dataframe(
    state_display.sort_values(
        "Portfolio Weight",
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)


fig_prior = go.Figure()

fig_prior.add_trace(
    go.Bar(
        x=state_df["Ticker"],
        y=state_df["BL Prior"] * 100,
        name="BL Prior",
    )
)

fig_prior.add_trace(
    go.Bar(
        x=state_df["Ticker"],
        y=state_df["BL Posterior"] * 100,
        name="BL Posterior",
    )
)

fig_prior.update_layout(
    title="Market-implied prior vs ML-updated posterior",
    yaxis_title="Expected annual return (%)",
    barmode="group",
    height=500,
)

st.plotly_chart(
    fig_prior,
    use_container_width=True,
)


# ======================================================================================
# 3. Risk diagnostics
# ======================================================================================

st.header("3. Risk Model Diagnostics")

r1, r2, r3, r4 = st.columns(4)

r1.metric(
    "Portfolio-risk window",
    f"{RISK_WINDOW_DAYS} days",
)

r2.metric(
    "BL covariance window",
    f"{BL_WINDOW_DAYS} days",
)

r3.metric(
    "Risk covariance LW shrinkage",
    f"{risk_shrinkage:.3f}",
)

r4.metric(
    "BL covariance LW shrinkage",
    f"{bl_shrinkage:.3f}",
)

risk_std = np.sqrt(
    np.diag(
        risk_cov.values
    )
)

risk_corr = (
    risk_cov.values
    / np.outer(
        risk_std,
        risk_std,
    )
)

fig_corr = go.Figure(
    data=go.Heatmap(
        z=risk_corr,
        x=risk_cov.columns,
        y=risk_cov.index,
        colorscale="RdBu_r",
        zmid=0,
    )
)

fig_corr.update_layout(
    title="Current portfolio-risk correlation matrix",
    height=600,
)

st.plotly_chart(
    fig_corr,
    use_container_width=True,
)


# ======================================================================================
# 4. Corrected walk-forward comparison
# ======================================================================================

st.header("4. Leakage-Safe Walk-Forward Comparison")

results = pd.DataFrame(
    {
        "Strategy": [
            "Equal Weight",
            "Sample-Cov Markowitz",
            "Shrinkage-Cov Markowitz",
            "Minimum Variance",
            "BL Prior Only",
            "BL + ML (uncertainty=1)",
        ],
        "Total Return": [
            2.1483,
            1.3775,
            1.3437,
            1.4241,
            1.4883,
            1.7635,
        ],
        "Annualized Return": [
            0.1952,
            0.1441,
            0.1416,
            0.1476,
            0.1522,
            0.1712,
        ],
        "Annualized Vol": [
            0.1689,
            0.1911,
            0.1925,
            0.1460,
            0.1542,
            0.1618,
        ],
        "Sharpe": [
            0.7707,
            0.4140,
            0.3977,
            0.5656,
            0.5659,
            0.6563,
        ],
        "Sortino": [
            0.8808,
            0.5041,
            0.4824,
            0.7135,
            0.6879,
            0.7927,
        ],
        "Max Drawdown": [
            -0.3457,
            -0.3387,
            -0.3383,
            -0.2593,
            -0.2795,
            -0.2783,
        ],
    }
)

display_results = results.copy()

for col in [
    "Annualized Return",
    "Annualized Vol",
    "Max Drawdown",
]:
    display_results[col] = (
        display_results[col]
        * 100
    ).map(
        lambda x: f"{x:.2f}%"
    )

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "The corrected 21-day ML model has mean OOS IC "
    f"{ML_OOS_IC:.4f}. Equal-weight remains the strongest Sharpe baseline, "
    "while BL + ML improves modestly over the BL prior-only portfolio."
)

fig_sharpe = px.bar(
    results,
    x="Strategy",
    y="Sharpe",
    title="Leakage-safe walk-forward Sharpe comparison",
)

st.plotly_chart(
    fig_sharpe,
    use_container_width=True,
)


# ======================================================================================
# 5. Leakage-safe confidence sensitivity
# ======================================================================================

st.header("5. BL View-Confidence Sensitivity")

confidence_results = pd.DataFrame(
    {
        "Uncertainty Multiplier": [
            1,
            2,
            3,
            5,
            10,
            15,
        ],
        "Total Return": [
            1.7635,
            1.6618,
            1.5743,
            1.5600,
            1.5470,
            1.5220,
        ],
        "Annualized Return": [
            0.1712,
            0.1644,
            0.1584,
            0.1574,
            0.1564,
            0.1547,
        ],
        "Annualized Vol": [
            0.1618,
            0.1598,
            0.1587,
            0.1576,
            0.1560,
            0.1552,
        ],
        "Sharpe": [
            0.6563,
            0.6221,
            0.5881,
            0.5859,
            0.5861,
            0.5777,
        ],
        "Sortino": [
            0.7927,
            0.7498,
            0.7100,
            0.7091,
            0.7096,
            0.7001,
        ],
        "Max Drawdown": [
            -0.2783,
            -0.2747,
            -0.2744,
            -0.2734,
            -0.2740,
            -0.2757,
        ],
    }
)

confidence_display = confidence_results.copy()

for col in [
    "Annualized Return",
    "Annualized Vol",
    "Max Drawdown",
]:
    confidence_display[col] = (
        confidence_display[col]
        * 100
    ).map(
        lambda x: f"{x:.2f}%"
    )

st.dataframe(
    confidence_display,
    use_container_width=True,
    hide_index=True,
)

fig_confidence = px.line(
    confidence_results,
    x="Uncertainty Multiplier",
    y="Sharpe",
    markers=True,
    title="ML-view uncertainty vs realized Sharpe",
)

fig_confidence.add_hline(
    y=0.5659,
    line_dash="dash",
    annotation_text="BL Prior Only = 0.5659",
)

fig_confidence.add_hline(
    y=0.7707,
    line_dash="dot",
    annotation_text="Equal Weight = 0.7707",
)

st.plotly_chart(
    fig_confidence,
    use_container_width=True,
)

st.info(
    "Lower uncertainty gives the ML views more influence. "
    "Sharpe rises from 0.5659 for the BL prior-only portfolio to 0.6563 "
    "at uncertainty multiplier 1. Multiplier 1 is used as the final system "
    "configuration because it was the strongest observed setting in the "
    "completed leakage-safe confidence sweep; it should be interpreted as "
    "the best tested setting, not as a universally optimal confidence level."
)


# ======================================================================================
# 6. Final configuration
# ======================================================================================

st.header("6. Final Strategy Configuration")

configuration_df = pd.DataFrame(
    {
        "Parameter": [
            "ML target horizon",
            "ML training purge",
            "Portfolio risk aversion",
            "Maximum stock weight",
            "Market risk aversion δ",
            "Prior uncertainty τ",
            "Relative ML views",
            "View uncertainty multiplier",
            "Portfolio-risk covariance window",
            "BL equilibrium covariance window",
            "Transaction cost",
            "Turnover cap",
        ],
        "Value": [
            "21 trading days",
            "21 trading dates before every rebalance",
            "3",
            "15%",
            "2.5",
            "0.05",
            "10",
            "1",
            "252 trading days",
            "756 trading days",
            "10 bps per unit turnover",
            "0.5 L1 turnover per rebalance",
        ],
    }
)

st.dataframe(
    configuration_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Research and educational use only — not investment advice."
)

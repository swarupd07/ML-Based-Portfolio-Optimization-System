# Streamlit UI for the current 21-day + Black-Litterman portfolio system

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_pipeline import load_cached_prices, engineer_features, build_long_dataset
from src.covariance import shrinkage_covariance
from src.optimizer import optimize_portfolio, portfolio_stats
from src.ml_predict import train_predict
from src.black_litterman import build_bl_expected_returns, implied_equilibrium_returns
from src.market_data import align_shares_to_prices, build_market_caps, get_market_weights_asof

st.set_page_config(
    page_title="Black-Litterman Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

st.title("Black-Litterman Portfolio Optimizer")
st.caption(
    "21-day ML views + market-implied Black-Litterman expected returns + "
    "Ledoit-Wolf risk estimation + constrained Markowitz allocation."
)

ROOT = Path(__file__).resolve().parent.parent
PRICES_PATH = ROOT / "data" / "prices.parquet"
SHARES_PATH = ROOT / "data" / "shares_outstanding.csv"

# Sidebar
st.sidebar.header("Portfolio settings")

investment_amount = st.sidebar.number_input(
    "Investment amount (₹)",
    min_value=10_000,
    value=1_000_000,
    step=10_000,
)

# Final research configuration used for the portfolio shown at the top.
# These are fixed so the displayed allocation corresponds to the same
# methodology whose completed walk-forward backtest produced Sharpe 0.8305.
portfolio_risk_aversion = 3.0
max_weight = 0.15
market_risk_aversion = 2.5
tau = 0.05
n_views = 10
uncertainty_multiplier = 3.0
risk_window_days = 252
bl_window_days = 756

st.sidebar.caption(
    "The top portfolio uses the final validated BL + ML research configuration."
)

@st.cache_data(show_spinner=False)
def load_project_data():
    if not PRICES_PATH.exists():
        raise FileNotFoundError("Missing data/prices.parquet.")

    prices = load_cached_prices().dropna(axis=1, how="all")
    features = engineer_features(prices)
    long_df = build_long_dataset(features)
    return prices, long_df

@st.cache_data(show_spinner=False)
def load_market_caps(prices):
    if not SHARES_PATH.exists():
        raise FileNotFoundError(
            "Missing data/shares_outstanding.csv. Run python download_market_data.py."
        )

    shares_df = pd.read_csv(SHARES_PATH)
    aligned = align_shares_to_prices(shares_df, prices)
    return build_market_caps(aligned, prices)

try:
    prices, long_df = load_project_data()
    market_caps = load_market_caps(prices)
except Exception as exc:
    st.error(str(exc))
    st.stop()

daily_returns = prices.pct_change()
latest_price_date = prices.index[-1]
latest_feature_date = pd.to_datetime(long_df["date"]).max()

st.caption(
    f"Data: **{len(prices.columns)} NSE large-cap stocks** | "
    f"{prices.index[0].date()} → {latest_price_date.date()} | "
    f"Latest usable ML feature date: **{latest_feature_date.date()}**"
)

# Build current model
risk_window = (
    daily_returns[daily_returns.index < latest_price_date]
    .tail(risk_window_days)
    .dropna(axis=1, how="any")
)

bl_window = (
    daily_returns[daily_returns.index < latest_price_date]
    .tail(bl_window_days)
)

min_obs = int(0.95 * len(bl_window))
bl_window = bl_window.dropna(axis=1, thresh=min_obs).ffill().dropna()

tickers = risk_window.columns.intersection(bl_window.columns)

if len(tickers) < 5:
    st.error("Too few assets have sufficient data for the selected windows.")
    st.stop()

risk_window = risk_window[tickers]
bl_window = bl_window[tickers]

risk_cov, risk_shrinkage = shrinkage_covariance(risk_window)
bl_cov, bl_shrinkage = shrinkage_covariance(bl_window)

market_weights = get_market_weights_asof(
    market_caps=market_caps,
    rebalance_date=latest_price_date,
    tickers=tickers,
)

ml_predictions, ml_diag = train_predict(
    long_df=long_df,
    train_end=latest_price_date,
    predict_date=latest_feature_date,
)

prior = implied_equilibrium_returns(
    cov_matrix=bl_cov,
    market_weights=market_weights,
    risk_aversion=market_risk_aversion,
)

bl_expected_returns = build_bl_expected_returns(
    cov_matrix=bl_cov,
    market_weights=market_weights,
    risk_aversion=market_risk_aversion,
    ml_predictions=ml_predictions,
    tau=tau,
    n_views=n_views,
    uncertainty_multiplier=uncertainty_multiplier,
)

weights = optimize_portfolio(
    expected_returns=bl_expected_returns,
    cov_matrix=risk_cov,
    risk_aversion=portfolio_risk_aversion,
    max_weight=max_weight,
)

stats = portfolio_stats(
    weights,
    bl_expected_returns,
    risk_cov,
)

# 1. Current allocation
st.header("1. Final BL + ML Portfolio Allocation")

st.success(
    "Final portfolio generated using the selected research configuration: "
    "21-day ML relative views, Black-Litterman posterior, 756-day BL covariance, "
    "252-day portfolio-risk covariance, δ=2.5, τ=0.05, 10 views, "
    "uncertainty multiplier=3, and portfolio risk aversion=3."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Walk-forward Sharpe",
    "0.8305",
    help="Realized out-of-sample Sharpe for this completed BL + ML configuration."
)
m2.metric(
    "Walk-forward annual return",
    "19.43%",
)
m3.metric(
    "Walk-forward volatility",
    "15.57%",
)
m4.metric(
    "Walk-forward max drawdown",
    "-25.80%",
)

st.caption(
    "The performance metrics above are historical walk-forward results for the "
    "same fixed BL + ML configuration. The allocation below is the latest portfolio "
    "generated by that methodology using the current posterior."
)

alloc = weights[weights > 0.001].sort_values(ascending=False)

allocation_df = pd.DataFrame({
    "Ticker": alloc.index,
    "Weight": alloc.values,
    "Weight %": alloc.values * 100,
    "Amount (₹)": alloc.values * investment_amount,
    "BL Expected Return": bl_expected_returns.reindex(alloc.index).values,
})

display_alloc = allocation_df.copy()
display_alloc["Weight %"] = display_alloc["Weight %"].map(lambda x: f"{x:.2f}%")
display_alloc["Amount (₹)"] = display_alloc["Amount (₹)"].map(lambda x: f"₹{x:,.0f}")
display_alloc["BL Expected Return"] = (
    display_alloc["BL Expected Return"] * 100
).map(lambda x: f"{x:.2f}%")
display_alloc = display_alloc.drop(columns=["Weight"])

a, b = st.columns([1, 1])

with a:
    fig_alloc = px.pie(
        allocation_df,
        names="Ticker",
        values="Weight",
        title="Current portfolio weights",
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

with b:
    st.dataframe(
        display_alloc,
        use_container_width=True,
        hide_index=True,
        height=500,
    )


with st.expander("Current model snapshot", expanded=False):
    s1, s2, s3 = st.columns(3)
    s1.metric("Current model-implied return", f"{stats['expected_return'] * 100:.2f}%")
    s2.metric("Current model-implied volatility", f"{stats['volatility'] * 100:.2f}%")
    s3.metric(
        "Current ex-ante Sharpe",
        f"{stats['sharpe_ratio']:.2f}",
        help=(
            "This is a current model estimate and can differ materially from the "
            "historical walk-forward Sharpe of 0.8305."
        ),
    )

# 2. BL state
st.header("2. Current Black-Litterman Model State")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean BL prior", f"{prior.mean() * 100:.2f}%")
c2.metric("Mean BL posterior", f"{bl_expected_returns.mean() * 100:.2f}%")
c3.metric("Largest BL posterior", f"{bl_expected_returns.max() * 100:.2f}%")
c4.metric("ML training rows", f"{ml_diag.get('n_train_rows', 0):,}")

state_df = pd.DataFrame({
    "Ticker": tickers,
    "Market Weight": market_weights.reindex(tickers).values,
    "ML 21d Prediction": ml_predictions.reindex(tickers).values,
    "BL Prior": prior.reindex(tickers).values,
    "BL Posterior": bl_expected_returns.reindex(tickers).values,
    "Portfolio Weight": weights.reindex(tickers).values,
})

state_display = state_df.copy()
for col in [
    "Market Weight",
    "ML 21d Prediction",
    "BL Prior",
    "BL Posterior",
    "Portfolio Weight",
]:
    state_display[col] = (
        state_display[col] * 100
    ).map(lambda x: f"{x:.2f}%")

st.dataframe(
    state_display.sort_values("Portfolio Weight", ascending=False),
    use_container_width=True,
    hide_index=True,
)

fig_prior = go.Figure()
fig_prior.add_trace(go.Bar(
    x=state_df["Ticker"],
    y=state_df["BL Prior"] * 100,
    name="BL Prior",
))
fig_prior.add_trace(go.Bar(
    x=state_df["Ticker"],
    y=state_df["BL Posterior"] * 100,
    name="BL Posterior",
))
fig_prior.update_layout(
    title="BL prior vs posterior",
    yaxis_title="Expected annual return (%)",
    barmode="group",
    height=500,
)
st.plotly_chart(fig_prior, use_container_width=True)

# 3. Risk diagnostics
st.header("3. Risk Model Diagnostics")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Portfolio-risk window", f"{risk_window_days} days")
r2.metric("BL covariance window", f"{bl_window_days} days")
r3.metric("Risk covariance LW shrinkage", f"{risk_shrinkage:.3f}")
r4.metric("BL covariance LW shrinkage", f"{bl_shrinkage:.3f}")

risk_std = np.sqrt(np.diag(risk_cov.values))
risk_corr = risk_cov.values / np.outer(risk_std, risk_std)

fig_corr = go.Figure(data=go.Heatmap(
    z=risk_corr,
    x=risk_cov.columns,
    y=risk_cov.index,
    colorscale="RdBu_r",
    zmid=0,
))
fig_corr.update_layout(
    title="Current portfolio-risk correlation matrix",
    height=600,
)
st.plotly_chart(fig_corr, use_container_width=True)

# 4. Current completed results
st.header("4. Walk-Forward Backtest Results")

st.caption(
    "These are realized out-of-sample results from the current 21-day-target "
    "pipeline. They are not dynamically recomputed when sidebar parameters change."
)

baseline_results = pd.DataFrame({
    "Strategy": [
        "Equal Weight",
        "Sample Covariance",
        "Shrinkage Covariance",
        "Minimum Variance",
    ],
    "Total Return": [2.1483, 2.1180, 2.0532, 1.4241],
    "Annualized Return": [0.1952, 0.1934, 0.1895, 0.1476],
    "Annualized Vol": [0.1689, 0.1892, 0.1912, 0.1460],
    "Sharpe": [0.7707, 0.6786, 0.6512, 0.5656],
    "Sortino": [0.8808, 0.8181, 0.7830, 0.7135],
    "Max Drawdown": [-0.3457, -0.3151, -0.3184, -0.2593],
})

bl_results = pd.DataFrame({
    "Strategy": [
        "BL + ML (uncertainty=3)",
        "BL Prior Only",
    ],
    "Total Return": [2.1335, 1.4883],
    "Annualized Return": [0.1943, 0.1522],
    "Annualized Vol": [0.1557, 0.1542],
    "Sharpe": [0.8305, 0.5659],
    "Sortino": [1.0303, 0.6879],
    "Max Drawdown": [-0.2580, -0.2795],
})

results = pd.concat([baseline_results, bl_results], ignore_index=True)
display_results = results.copy()

for col in ["Annualized Return", "Annualized Vol", "Max Drawdown"]:
    display_results[col] = (
        display_results[col] * 100
    ).map(lambda x: f"{x:.2f}%")

st.dataframe(display_results, use_container_width=True, hide_index=True)

b1, b2, b3, b4 = st.columns(4)
b1.metric("BL + ML Sharpe", "0.8305")
b2.metric("Equal-weight Sharpe", "0.7707")
b3.metric("BL + ML volatility", "15.57%")
b4.metric("BL + ML max drawdown", "-25.80%")

fig_sharpe = px.bar(
    results,
    x="Strategy",
    y="Sharpe",
    title="Walk-forward Sharpe comparison",
)
st.plotly_chart(fig_sharpe, use_container_width=True)

# 5. Confidence sensitivity
st.header("5. BL View-Confidence Sensitivity")

confidence_results = pd.DataFrame({
    "Uncertainty Multiplier": [1, 2, 3, 4, 5, 10, 15],
    "Annualized Return": [0.1972, 0.1968, 0.1943, 0.1933, 0.1913, 0.1834, 0.1752],
    "Annualized Vol": [0.1596, 0.1569, 0.1557, 0.1551, 0.1549, 0.1544, 0.1542],
    "Sharpe": [0.8288, 0.8397, 0.8305, 0.8270, 0.8155, 0.7672, 0.7150],
    "Sortino": [1.0338, 1.0453, 1.0303, 1.0223, 1.0067, 0.9415, 0.8773],
    "Max Drawdown": [-0.2606, -0.2589, -0.2580, -0.2587, -0.2606, -0.2677, -0.2707],
})

conf_display = confidence_results.copy()
for col in ["Annualized Return", "Annualized Vol", "Max Drawdown"]:
    conf_display[col] = (
        conf_display[col] * 100
    ).map(lambda x: f"{x:.2f}%")

st.dataframe(conf_display, use_container_width=True, hide_index=True)

fig_conf = px.line(
    confidence_results,
    x="Uncertainty Multiplier",
    y="Sharpe",
    markers=True,
    title="View uncertainty vs realized Sharpe",
)
fig_conf.add_hline(y=0.7707, line_dash="dot", annotation_text="Equal weight = 0.7707")
fig_conf.add_hline(y=0.5659, line_dash="dash", annotation_text="BL prior only = 0.5659")
st.plotly_chart(fig_conf, use_container_width=True)

st.caption(
    "Multiplier 2 produced the highest observed Sharpe (0.8397), but 3 remains "
    "the default moderate-confidence setting to avoid presenting a test-period "
    "maximum as a universally optimal parameter."
)

# 6. Final strategy configuration
st.header("6. Final Strategy Configuration")

configuration_df = pd.DataFrame({
    "Parameter": [
        "ML forecast horizon",
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
        "3",
        "15%",
        "2.5",
        "0.05",
        "10",
        "3",
        "252 trading days",
        "756 trading days",
        "10 bps per unit turnover",
        "0.5 L1 turnover per rebalance",
    ],
})

st.dataframe(
    configuration_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Research and educational use only — not investment advice."
)

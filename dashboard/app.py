# Streamlit dashboard for the portfolio optimizer

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_pipeline import (
    synthetic_prices, load_cached_prices, engineer_features, build_long_dataset,
)
from src.covariance import sample_covariance, shrinkage_covariance, compare_estimators
from src.optimizer import optimize_portfolio, portfolio_stats
from src.ml_predict import train_predict, historical_mean_baseline

# ------------------------------------------------------------------------------------

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("ML-Based Portfolio Optimization")
st.caption(
    "Shrinkage-covariance Markowitz optimization with ML return tilts. "
    "Built for demonstration — not investment advice."
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "prices.parquet"

# --------------------------------------sidebar ---------------------------------------

st.sidebar.header("Investor inputs")
amount = st.sidebar.number_input("Investment amount (₹)", min_value=10_000, value=1_000_000, step=10_000)
risk_label = st.sidebar.select_slider(
    "Risk preference", options=["Aggressive", "Balanced", "Conservative"], value="Balanced"
)

risk_aversion_map = {"Conservative": 25.0, "Balanced": 15.0, "Aggressive": 8.0}
max_weight = st.sidebar.slider("Max weight per stock", 0.05, 0.35, 0.15, 0.05)

data_available = DATA_PATH.exists()
if data_available:
    use_synthetic = st.sidebar.checkbox("Use synthetic data instead (no internet needed)", value=False)
else:
    st.sidebar.warning(
        "No cached real data found at data/prices.parquet. Using synthetic data. "
        "Run:\n\n`python -m src.data_pipeline --tickers config/nifty50.txt --start 2019-01-01`\n\n"
        "then reload this page."
    )
    use_synthetic = True

# --------------------------------------- data -------------------------------------------

@st.cache_data
def load_data(synthetic: bool):
    if synthetic:
        prices = synthetic_prices()
    else:
        prices = load_cached_prices()
        # Dropping any all-NaN columns defensively 
        prices = prices.dropna(axis=1, how="all")
    feats = engineer_features(prices)
    long_df = build_long_dataset(feats)
    return prices, feats, long_df

try:
    prices, feats, long_df = load_data(use_synthetic)
except FileNotFoundError:
    st.error(
        "Could not load data/prices.parquet even though it appeared to exist. "
        "Try re-running the data pipeline."
    )
    st.stop()

returns = prices.pct_change().dropna()
latest_date = prices.index[-1]

source_label = "synthetic data" if use_synthetic else f"real data ({len(prices.columns)} tickers, {prices.index[0].date()} to {prices.index[-1].date()})"
st.caption(f"Currently showing: **{source_label}**")

# --------------------------------- covariance panel ------------------------------

st.header("1. Covariance estimation: sample vs. shrinkage")
col1, col2 = st.columns([1, 1])

samp_cov = sample_covariance(returns)
shrink_cov, delta = shrinkage_covariance(returns)

with col1:
    st.markdown(f"**Ledoit-Wolf shrinkage intensity:** `{delta:.3f}`")
    st.markdown(
        "This is how much the estimator pulled the noisy sample covariance "
        "toward a structured target — higher means the raw sample matrix was "
        "judged less trustworthy given the amount of data available. "
        "Shrinkage intensity rises as the estimation window shrinks relative "
        "to the number of assets (fewer observations per parameter to estimate)."
    )
    st.dataframe(compare_estimators(returns).round(4))

with col2:
    corr = samp_cov.copy()
    std = pd.Series(1.0, index=corr.index)
    for i in corr.index:
        std[i] = corr.loc[i, i] ** 0.5
    corr_matrix = corr.div(std, axis=0).div(std, axis=1)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=[t.replace("&", "&amp;") for t in corr_matrix.columns],
        y=[t.replace("&", "&amp;") for t in corr_matrix.index],
        colorscale="RdBu_r", zmid=0,
    ))
    fig.update_layout(title="Sample correlation structure", height=420)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------- ML tab ------------------------------ 

st.header("2. Expected return signal (ML vs. historical mean)")
last_available_date = long_df["date"].max()
ml_preds, diag = train_predict(long_df, train_end=latest_date, predict_date=last_available_date)
hist_mean = historical_mean_baseline(long_df, train_end=latest_date)

st.markdown(
    f"Trained on **{diag.get('n_train_rows', 0):,}** rows. "
    "In-sample training IC (rank correlation) shown below — "
    "**this is not the honest evaluation number.** The real, out-of-sample "
    "evaluation comes from walk-forward backtesting (`src/backtest.py`), "
    "which on this dataset showed a mean IC of **~0.015** — essentially no "
    "usable signal, which is the expected, honest result for daily stock "
    "return prediction, not a failure of the pipeline."
)
c1, c2 = st.columns(2)
c1.metric("Training IC (in-sample)", f"{diag.get('train_ic', float('nan')):.4f}")
c2.metric("Out-of-sample IC (from backtest)", "~0.015", help="From src/backtest.py walk-forward run")

# --------------------------------- optimizer tab ----------------------------------

st.header("3. Recommended allocation")

expected_returns = hist_mean.reindex(prices.columns).fillna(0) + 5.0 * ml_preds.reindex(prices.columns).fillna(0)
weights = optimize_portfolio(
    expected_returns, shrink_cov,
    risk_aversion=risk_aversion_map[risk_label], max_weight=max_weight,
)
stats = portfolio_stats(weights, expected_returns, shrink_cov)

c1, c2, c3 = st.columns(3)
c1.metric("Expected annual return", f"{stats['expected_return']*100:.1f}%")
c2.metric("Expected volatility", f"{stats['volatility']*100:.1f}%")
c3.metric("Sharpe ratio", f"{stats['sharpe_ratio']:.2f}")

alloc = weights[weights > 0.005].sort_values(ascending=False)
alloc_df = (alloc * amount).round(0).reset_index()
alloc_df.columns = ["Ticker", "Amount (₹)"]
alloc_df["Weight %"] = (alloc.values * 100).round(1)
alloc_df["Ticker"] = alloc_df["Ticker"].str.replace("&", "&amp;")

fig_pie = px.pie(alloc_df, names="Ticker", values="Amount (₹)", title="Recommended allocation")
col_a, col_b = st.columns([1, 1])
col_a.plotly_chart(fig_pie, use_container_width=True)
col_b.dataframe(alloc_df, use_container_width=True, hide_index=True)

# --------------------------------- key finding -----------------------------

st.header("4. Key finding: does optimization actually beat naive diversification?")
st.markdown(
    "**Short answer: not by default.** In a full walk-forward backtest on this "
    "dataset (39 NSE large-caps, monthly rebalancing, 10bps transaction costs "
    "per unit turnover), naive equal-weight outperformed Markowitz optimization "
    "at low risk-aversion — a real, documented phenomenon known as the "
    "**1/N puzzle** (DeMiguel, Garlappi & Uppal, 2009)."
)

finding_df = pd.DataFrame({
    "Strategy": [
        "Equal-weight (1/N)",
        "Minimum-variance",
        "Markowitz, risk_aversion=25",
        "Markowitz, risk_aversion=3 (naive default)",
    ],
    "Sharpe Ratio": [0.79, 0.60, 0.59, 0.26],
})
st.dataframe(finding_df, hide_index=True, use_container_width=True)

st.markdown(
    "**Root cause, isolated by ablation:** the bottleneck was not the "
    "covariance estimator — sample vs. shrinkage covariance produced nearly "
    "identical Sharpe ratios (0.284 vs. 0.258) at the default lookback window. "
    "The bottleneck was **estimation error in expected returns**. Historical "
    "mean returns are extremely noisy estimators over realistic backtest "
    "windows, and mean-variance optimization amplifies that noise into "
    "confident, wrong bets. Raising `risk_aversion` — which down-weights how "
    "much the optimizer trusts the noisy return estimate — recovered "
    "performance and converged toward the minimum-variance limit as expected "
    "from the underlying math. This dashboard's 'Balanced' setting now "
    "reflects that finding (risk_aversion=15) rather than the naive first "
    "guess (risk_aversion=3) used before this was tested."
)

st.info(
    "Run `python run_real_backtest.py` for the full walk-forward comparison "
    "with transaction costs — that backtest, not this single snapshot "
    "allocation, is the actual evidence for how each strategy performs."
)

# ------------------------------------------------------------------------------------
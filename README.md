# ML-Based Portfolio Optimization System

An end-to-end quantitative investment platform that predicts stock returns using
machine learning, estimates portfolio risk using **Ledoit-Wolf shrinkage
covariance estimation**, and allocates capital via **convex (Markowitz)
optimization** — validated with an honest, walk-forward backtest that includes
transaction costs.

This README documents not just what the system does, but **what we actually
found when we tested it on real NSE data** — including a result that
contradicted the initial hypothesis, and the investigation that explained why.

---

## Table of Contents

1. [Why this project](#1-why-this-project)
2. [System architecture](#2-system-architecture)
3. [Methodology](#3-methodology)
4. [Experiments and results](#4-experiments-and-results)
5. [Key finding: the 1/N puzzle](#5-key-finding-the-1n-puzzle)
6. [Project structure](#6-project-structure)
7. [How to run](#7-how-to-run)
8. [Honesty checklist](#8-honesty-checklist)
9. [Limitations and future work](#9-limitations-and-future-work)
10. [Tech stack](#10-tech-stack)

---

## 1. Why this project

I wanted to build something that combined the math I'd been studying —
linear algebra, statistics, optimization — with a real, honestly-evaluated
system, rather than a notebook that predicts stock prices and stops there.
Two principles guided how I approached it:

- **Predicting individual stock returns is genuinely very hard.** A
  suspiciously good R² almost always means the model is leaking future
  information or overfitting, not finding real alpha — so I set out to
  measure and report signal quality honestly (via Information Coefficient)
  rather than optimize for a flattering number.
- **"I ran an optimizer" isn't a finding on its own.** Without honest
  backtesting against real baselines, there's no way to know if the
  optimization actually adds value or is quietly *destroying* value through
  overfitting and transaction costs — so the project is built around a
  strict walk-forward backtest from the start, not added afterward to
  validate a result I already believed.

I focused the project on one specific, well-known technical problem in
quantitative finance:

> With `N` assets and `T` return observations, the sample covariance matrix
> has `N(N+1)/2` free parameters to estimate. When `N` is large relative to
> `T`, the sample covariance is noisy and poorly conditioned — and because
> Markowitz optimization inverts this matrix, **estimation error gets
> amplified**, not averaged out. This is a real, published problem
> (Ledoit & Wolf, 2004) that I wanted to address directly rather than
> ignore.

The plan was to prove that **shrinkage covariance estimation** produces a
more stable, better-performing portfolio than the naive sample covariance
most tutorials use. Along the way, real testing surfaced a more interesting
and more important finding than the original hypothesis — documented in
[Section 5](#5-key-finding-the-1n-puzzle).

---

## 2. System architecture

```
Historical prices (yfinance, NSE large-caps)
        │
        ▼
Feature engineering (returns, momentum, volatility, RSI, MACD)
        │
        ├──────────────────► Covariance estimation
        │                     (sample vs. Ledoit-Wolf shrinkage)
        ▼
ML expected-return model (LightGBM, walk-forward trained)
        │
        ▼
Markowitz optimizer (cvxpy: maximize return − risk_aversion × risk)
        │
        ▼
Walk-forward backtest (monthly rebalancing, transaction costs, turnover cap)
        │
        ▼
Compare: Equal-weight | Sample-cov Markowitz | Shrinkage-cov Markowitz | Min-variance
        │
        ▼
Streamlit dashboard
```

Every stage is designed around one non-negotiable rule: **no lookahead.** At
every rebalance date, the ML model, the covariance matrix, and the expected
return estimate use only data strictly before that date. This is what makes
the backtest results trustworthy rather than optimistic fiction.

---

## 3. Methodology

### 3.1 Data

- **Universe:** 39 NSE large-cap stocks (originally 40 — `TATAMOTORS.NS` was
  dropped after its October 2025 corporate demerger split it into
  `TMPV.NS` and `TMCV.NS`, breaking return continuity across the backtest
  window; see [Section 8](#8-honesty-checklist)).
- **Period:** 2019-01-01 to present (~1,880 trading days, ~7.5 years).
- **Source:** Yahoo Finance via `yfinance`, adjusted close prices
  (`auto_adjust=True`, so splits/dividends are handled automatically).

### 3.2 Features (per stock, computed causally — no future data used)

| Feature | Description |
|---|---|
| `mom_5d`, `mom_21d` | 5-day and 21-day price momentum |
| `vol_21d`, `vol_63d` | Rolling realized volatility |
| `ma_ratio_10_50` | 10-day vs. 50-day moving average ratio |
| `rsi_14` | 14-day Relative Strength Index |
| `macd_hist` | MACD histogram (12/26/9) |
| `fwd_return_5d` | **Target only** — forward 5-day return, shifted to avoid lookahead |

### 3.3 Covariance estimation — the technical centerpiece

Two estimators are computed and compared at every rebalance:

- **Sample covariance** — the textbook estimator, `returns.cov()`. Unbiased
  but high-variance, especially as the number of assets grows relative to
  the number of observations.
- **Ledoit-Wolf shrinkage** — `Σ_shrunk = δ·F + (1−δ)·Σ_sample`, where `F` is
  a structured, low-variance target and `δ` (the shrinkage intensity) is
  chosen analytically via the closed-form Ledoit-Wolf (2004) estimator — not
  a hand-tuned hyperparameter.

**Diagnostic used to compare them:** the *condition number* (ratio of
largest to smallest eigenvalue). A high condition number means the matrix is
close to singular, so small input noise gets massively amplified when
Markowitz optimization inverts it.

### 3.4 Expected return estimation

Two sources, blended:

- **Historical mean** — trailing annualized mean daily return. The simple,
  robust baseline.
- **ML tilt** — a LightGBM regressor trained walk-forward on the engineered
  features, predicting 5-day forward returns. Its signal quality is measured
  honestly via **Information Coefficient (IC)** — the Spearman rank
  correlation between predicted and realized returns — rather than R²,
  which is nearly meaningless for return prediction at this noise level.

### 3.5 Optimization

A convex mean-variance optimizer (`cvxpy`, `OSQP` solver) solves:

```
maximize    w'μ − risk_aversion · w'Σw
subject to  Σw = 1
            0 ≤ w_i ≤ max_weight   (per-stock cap, default 15%)
            ‖w − w_prev‖₁ ≤ turnover_limit   (optional, caps rebalancing cost)
```

`risk_aversion` is the key lever: low values trust the return estimate `μ`
heavily; high values effectively ignore it and behave like a pure
minimum-variance portfolio.

### 3.6 Backtesting

Strict walk-forward validation: at every monthly rebalance date, the model is
retrained and the covariance matrix recomputed using **only data strictly
before that date**. Transaction costs (10 bps per unit of turnover) and a
turnover cap are applied — a strategy that looks great on paper but ignores
trading costs is not a real strategy.

Four strategies are run side by side every rebalance:

| Strategy | What it uses |
|---|---|
| Equal-weight (1/N) | Nothing — the naive baseline |
| Sample-cov Markowitz | Naive covariance, blended expected returns |
| Shrinkage-cov Markowitz | Ledoit-Wolf covariance, blended expected returns |
| Minimum-variance | Shrinkage covariance only, ignores expected returns entirely |

---

## 4. Experiments and results

### 4.1 Does shrinkage improve covariance conditioning? (Yes — and the effect scales with data scarcity)

At the full ~1,880-day history, shrinkage's advantage over sample covariance
was modest:

| Metric | Sample | Ledoit-Wolf (δ=0.022) |
|---|---|---|
| Condition number | 60.46 | 55.04 |
| Mean abs. off-diagonal correlation | 0.308 | 0.301 |
| Frobenius norm | 1.216 | 1.193 |

Only ~9% improvement in conditioning — because with 1,880 observations for
39 assets, the sample covariance is already reasonably well-estimated.

**We then tested the hypothesis properly** by repeating this comparison
across shorter lookback windows, to see whether shrinkage's value is
conditional on data scarcity:

| Window (days) | Shrinkage δ | Sample cond. # | Shrinkage cond. # | Improvement |
|---|---|---|---|---|
| 1,880 (full history) | 0.022 | 60.5 | 55.0 | ~9% |
| 756 (~3 years) | 0.036 | 55.2 | 47.1 | ~15% |
| 378 (~1.5 years) | 0.045 | 69.9 | 55.3 | ~21% |
| 189 (~9 months) | 0.075 | 109.5 | 64.7 | **~41%** |

**Finding:** shrinkage intensity rises monotonically as the estimation
window shrinks, and the condition-number gap between sample and shrinkage
widens sharply — from a 9% improvement at full history to a 41% improvement
at 189 days. This confirms the estimator is behaving exactly as theory
predicts: it correctly detects when it has less trustworthy data and leans
more on the structured target. **Shrinkage's value is conditional, not
constant — and it matters most exactly when a strategy would need to use
a short, recent lookback window** (which is common in practice, since
markets aren't stationary and stale multi-year correlation estimates can be
misleading).

### 4.2 Full walk-forward backtest (default config: `risk_aversion=3`, monthly rebalance, 10bps costs)

| Strategy | Annualized Return | Annualized Vol | **Sharpe** | Sortino | Max Drawdown |
|---|---|---|---|---|---|
| Equal-weight | 19.9% | 16.9% | **0.79** | 0.91 | −34.6% |
| Sample-cov Markowitz | 11.7% | 18.4% | 0.28 | 0.34 | −34.4% |
| Shrinkage-cov Markowitz | 11.3% | 18.5% | 0.26 | 0.31 | −34.7% |
| Min-variance | 15.2% | 14.6% | 0.60 | 0.75 | −25.9% |

**Mean out-of-sample IC: 0.015** — essentially no usable predictive signal,
reported honestly rather than hidden.

This was **not** the expected outcome. Naive equal-weight beat every
"sophisticated" strategy, including both Markowitz variants. This result
triggered a deeper investigation rather than being buried — see Section 5.

---

## 5. Key finding: the 1/N puzzle

### 5.1 Diagnosing the surprise

**Hypothesis 1 — is the ML signal actively hurting performance?**
Tested by removing the ML tilt entirely and using only the historical mean:

| Configuration | Sharpe |
|---|---|
| Historical mean + ML tilt | 0.258 |
| Historical mean only (no ML) | **0.186** — worse |

Removing the ML component made performance *worse*, not better. This ruled
out "the ML signal is pure noise dragging down performance" — despite its
tiny IC (0.015), it was mildly helping.

**Hypothesis 2 — is the covariance estimator the problem?**
Already ruled out by Section 4.2: sample-cov and shrinkage-cov Markowitz
produced nearly identical Sharpe ratios (0.284 vs. 0.258) — a difference
within noise, not a meaningful gap.

**Hypothesis 3 — is the *expected return* estimate the bottleneck?**
This is confirmed to be the real cause. Historical mean returns are
extremely noisy estimators over realistic backtest windows — a stock with
12% true annual return and 25% annual volatility needs *decades* of data
for its sample mean to be statistically distinguishable from zero.
Mean-variance optimization takes this noisy `μ` and makes **confident,
concentrated bets on it**, which is precisely how it amplifies estimation
error into poor real-world performance. This is a well-documented
phenomenon in the literature — the **"1/N puzzle"**
(DeMiguel, Garlappi & Uppal, *"Optimal Versus Naive Diversification"*,
2009) — which shows naive equal-weighting frequently beats "optimal"
mean-variance portfolios out-of-sample for exactly this reason.

### 5.2 Proving the mechanism: the risk_aversion sweep

If the noisy return estimate `μ` is the real problem, then an optimizer
that relies on it *less* should perform *better*. We tested this directly
by sweeping `risk_aversion` (which controls how heavily the optimizer
weighs `μ` vs. minimizing risk):

| risk_aversion | Sharpe |
|---|---|
| 3 (original default) | 0.186 |
| 8 | 0.435 |
| 15 | 0.569 |
| 25 | 0.594 |
| **Minimum-variance** (ignores `μ` completely) | **0.596** |

**This is the cleanest result in the project.** Sharpe rises *monotonically*
as reliance on the noisy return estimate decreases, and **converges almost
exactly to the minimum-variance portfolio's Sharpe** at high risk_aversion.
This is not a coincidence — it's a mathematical limit: as
`risk_aversion → ∞`, the `w'μ` term in the optimizer's objective becomes
negligible relative to the risk term, so the optimizer necessarily
degenerates into the minimum-variance solution. **This experiment
empirically demonstrates that limit**, which both validates that the
optimizer is behaving correctly and confirms the diagnosis: the return
estimate, not the covariance estimate, was the dominant source of error.

### 5.3 The honest conclusion

Even the best-tuned Markowitz configuration (Sharpe ≈ 0.594–0.596) still
did not beat naive equal-weight (Sharpe 0.79) on this dataset and window.
This is reported as-is rather than adjusted or cherry-picked, because the
investigation that explains *why* is more valuable than a single flattering
number:

> Contrary to the common assumption that better covariance estimation
> directly improves portfolio performance, the dominant driver of
> underperformance versus naive diversification was estimation error in
> the *expected return* vector, not the covariance matrix. Increasing
> risk-aversion — which down-weights reliance on the noisy mean estimate —
> recovered most of the performance gap and converged toward the
> theoretical minimum-variance limit, confirming the mechanism. This
> replicates a well-known result in the quantitative finance literature
> (DeMiguel, Garlappi & Uppal, 2009) using real NSE market data.

---

## 6. Project structure

```
portfolio-optimizer/
├── config/
│   └── nifty50.txt              # 39-ticker NSE large-cap universe
├── data/
│   ├── prices.parquet           # cached adjusted close prices
│   └── features_long.parquet    # engineered features, long format
├── src/
│   ├── data_pipeline.py         # download prices, engineer features
│   ├── covariance.py            # sample vs. Ledoit-Wolf shrinkage estimators
│   ├── ml_predict.py            # LightGBM expected-return model
│   ├── optimizer.py             # cvxpy Markowitz optimizer with constraints
│   └── backtest.py              # walk-forward backtest engine + metrics
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── run_real_backtest.py          # reproduces Section 4.2 results
├── run_ablation.py               # reproduces Section 5.1 ablation
└── requirements.txt
```

---

## 7. How to run

```bash
pip install -r requirements.txt

# 1. Pull real data (needs internet access to Yahoo Finance)
python -m src.data_pipeline --tickers config/nifty50.txt --start 2019-01-01

# 2. Run the full walk-forward backtest (reproduces Section 4.2)
python run_real_backtest.py

# 3. Run the ablation study (reproduces Section 5.1)
python run_ablation.py

# 4. Launch the interactive dashboard
streamlit run dashboard/app.py
```

The dashboard defaults to real cached data if `data/prices.parquet` exists,
and falls back to synthetic data (with a clear on-screen notice) otherwise —
so it never silently shows misleading numbers.

---

## 8. Honesty checklist

Things this project deliberately does **not** fake or hide:

- **The ML model's real out-of-sample IC (0.015) is reported prominently**,
  including in the dashboard itself, right next to the much higher but
  meaningless in-sample training IC (0.149) — so the overfitting gap is
  visible, not buried.
- **The backtest is strictly walk-forward.** No stage of the pipeline ever
  sees data from after the date it's making a decision for.
- **Transaction costs and a turnover cap are applied** to every strategy,
  including the baselines.
- **A real corporate action broke the data pipeline mid-project**
  (`TATAMOTORS.NS` → `TMPV.NS`/`TMCV.NS` demerger, October 2025) and was
  handled by dropping the ticker with an explicit, documented rationale,
  rather than silently patching around it.
- **The headline result contradicts the original hypothesis.** Naive
  equal-weight beat every optimized strategy tested. Rather than adjusting
  parameters until a flattering number appeared, the discrepancy was
  investigated and explained (Section 5).

---

## 9. Limitations and future work

- **Universe survivorship bias:** the 39-stock universe is today's NSE
  large-caps, not the large-cap universe as it existed in 2019 — this
  overstates historical performance somewhat, a known limitation of
  backtests that don't use point-in-time index membership.
- **No benchmark against a market index** (e.g., Nifty 50 itself) is
  currently included — a natural next addition, since "how does this
  compare to just buying the index" is the most common real-world
  question this project doesn't yet answer directly.
- **Black-Litterman as a natural next step:** the 1/N puzzle finding
  suggests the real fix isn't better covariance estimation but better
  *return* estimation — Black-Litterman blends market-implied equilibrium
  returns with investor views via Bayesian updating, directly addressing
  the noisy-mean problem this project identified, rather than treating
  historical averages as ground truth.
- **Single-country, single-asset-class scope:** extending to multi-asset
  (bonds, gold, international equities) would test whether the 1/N puzzle
  finding holds when the asset universe is more heterogeneous.

---

## 10. Tech stack

- **Data:** `yfinance`, `pandas`, `pyarrow`
- **Statistics/ML:** `numpy`, `scikit-learn` (Ledoit-Wolf, gradient boosting
  fallback), `lightgbm`, `scipy`
- **Optimization:** `cvxpy` (OSQP solver)
- **Dashboard:** `streamlit`, `plotly`

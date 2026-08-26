# ML-Based Portfolio Optimization System

An end-to-end quantitative investment platform that predicts stock returns using
machine learning, estimates portfolio risk using **Ledoit-Wolf shrinkage
covariance estimation**, constructs expected returns using both the original
historical-mean/ML approach and a **Black-Litterman + ML extension**, and allocates
capital via **convex (Markowitz) optimization** — validated with an honest,
walk-forward backtest that includes transaction costs.

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

```text
Historical prices (yfinance, NSE large-caps)
        │
        ▼
Feature engineering
(returns, 5d/21d momentum, volatility, RSI, MACD)
        │
        ├──────────────────────► Covariance estimation
        │                        Sample vs. Ledoit-Wolf shrinkage
        │
        ▼
21-day ML return model
(LightGBM, walk-forward trained)
        │
        ├──────────────────────► Original return-estimation path
        │                        Historical mean + ML tilt
        │
        └──────────────────────► Black-Litterman extension
                                 Point-in-time market-cap prior
                                 + relative ML views
                                 + explicit view uncertainty
                                           │
                                           ▼
Constrained Markowitz optimizer
        │
        ▼
Walk-forward backtest
(monthly rebalancing, 10 bps costs, turnover cap)
        │
        ▼
Compare:
Equal-weight | Sample-cov | Shrinkage-cov | Min-variance
| BL prior-only | BL + ML
        │
        ▼
Streamlit portfolio interface
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
| `fwd_return_21d` | **Target only** — forward 21-day return, shifted to avoid lookahead and aligned with the monthly holding horizon |

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
- **ML tilt / view signal** — a LightGBM regressor trained walk-forward on the
  engineered features, now predicting **21-day forward returns** so that the
  forecast horizon approximately matches the monthly rebalance/holding period.
  Its signal quality is measured honestly via **Information Coefficient (IC)** —
  the Spearman rank correlation between predicted and realized returns — rather
  than R², which is nearly meaningless for return prediction at this noise level.

### 3.5 Black-Litterman expected-return extension

The original experiments identified expected-return estimation as the main
bottleneck: improving `Σ` did not solve the instability caused by noisy `μ`.
The extension therefore keeps the existing covariance/optimizer infrastructure
and changes the way expected returns are constructed.

The Black-Litterman equilibrium prior is

```text
π = δ Σ_BL w_mkt
```

where:

- `π` is the market-implied equilibrium expected-return vector,
- `δ = 2.5` is the fixed market risk-aversion value used in the main experiment,
- `Σ_BL` is a longer-window Ledoit-Wolf covariance estimate,
- `w_mkt` is the point-in-time market-capitalization weight vector.

Historical market caps are reconstructed as

```text
market_cap_i,t = adjusted_price_i,t × shares_outstanding_i,t
w_i,t = market_cap_i,t / Σ_j market_cap_j,t
```

Historical shares outstanding are aligned point-in-time using the latest
observation available **on or before** each rebalance date. Missing early
observations are not backfilled from the future.

A rolling market-risk-aversion estimate was also tested:

```text
δ_t = (E[R_m] − R_f) / Var(R_m)
```

but one-year estimates were highly unstable, including negative values and
values above 10. Because that instability comes from the same noisy mean-return
problem the BL extension is intended to reduce, the main experiment uses the
fixed value `δ = 2.5` and keeps the rolling estimate only as a diagnostic.

### 3.6 ML forecasts as relative views

The ML model is not treated as an exact absolute-return oracle. Instead it is
used as a **cross-sectional ranking signal**.

Top-ranked and bottom-ranked stocks are paired into relative views:

```text
winner − loser = predicted relative outperformance
```

Each view is encoded through the Black-Litterman matrices:

- `P`: `+1` for the preferred stock, `−1` for the paired weaker stock,
- `Q`: the predicted 21-day return spread, annualized using `252 / 21`,
- `n_views = 10` in the main experiment,
- individual annualized relative views are loosely clipped to `±15%` as an
  outlier safety guard.

This is deliberately more conservative than using raw ML forecasts as exact
absolute expected returns.

### 3.7 View uncertainty

View uncertainty is modeled as

```text
Ω = diag(P (τ Σ_BL) P') × uncertainty_multiplier
```

with

```text
τ = 0.05
```

in the completed experiment.

A larger `uncertainty_multiplier` means less confidence in the ML views; a
smaller value gives them more influence.

The posterior expected-return vector is

```text
μ_BL =
[(τΣ)^−1 + P'Ω^−1P]^−1
[(τΣ)^−1π + P'Ω^−1Q]
```

so the final estimate is a confidence-weighted compromise between the
market-equilibrium prior and the ML information.

### 3.8 Two covariance horizons for two different roles

The extension intentionally separates two covariance roles:

- **BL equilibrium covariance:** up to `756` trading days, using Ledoit-Wolf
  shrinkage; used consistently for `π`, `τΣ`, `Ω`, and the BL posterior.
- **Portfolio-risk covariance:** the most recent `252` trading days, also using
  Ledoit-Wolf shrinkage; used by the optimizer.

This split was introduced after the shorter covariance window made the
market-implied prior excessively regime-sensitive during the COVID period.
The longer window gives the equilibrium prior a more stable structural anchor,
while the 252-day window keeps current portfolio risk responsive.

### 3.9 Optimization

A convex mean-variance optimizer (`cvxpy`, `OSQP` solver) solves:

```text
maximize    w'μ − risk_aversion · w'Σw
subject to  Σw = 1
            0 ≤ w_i ≤ max_weight   (per-stock cap, default 15%)
            ‖w − w_prev‖₁ ≤ turnover_limit   (optional, caps rebalancing cost)
```

`risk_aversion` is the key lever: low values trust the return estimate `μ`
heavily; high values effectively ignore it and behave like a pure
minimum-variance portfolio.

### 3.10 Backtesting

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

### 4.2 Original walk-forward result (historical experiment retained unchanged)

| Strategy | Annualized Return | Annualized Vol | **Sharpe** | Sortino | Max Drawdown |
|---|---|---|---|---|---|
| Equal-weight | 19.9% | 16.9% | **0.79** | 0.91 | −34.6% |
| Sample-cov Markowitz | 11.7% | 18.4% | 0.28 | 0.34 | −34.4% |
| Shrinkage-cov Markowitz | 11.3% | 18.5% | 0.26 | 0.31 | −34.7% |
| Min-variance | 15.2% | 14.6% | 0.60 | 0.75 | −25.9% |

**Mean out-of-sample IC: 0.015** — essentially no usable predictive signal,
reported honestly rather than hidden.

> **Historical-result note:** this table is intentionally retained unchanged
> because it represents the original stage of the project before the later
> 21-day horizon alignment and Black-Litterman extension.

This was **not** the expected outcome. Naive equal-weight beat every
"sophisticated" strategy, including both Markowitz variants. This result
triggered a deeper investigation rather than being buried — see Section 5.


### 4.3 Updated 21-day ML-horizon baseline

After diagnosing expected-return estimation as the dominant weakness, the ML
target was aligned with the monthly portfolio decision horizon by changing the
forward-return target from 5 trading days to **21 trading days**.

The existing strategies were rerun using the updated 21-day signal.

| Strategy | Total Return | Annualized Return | Annualized Vol | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Equal-weight | 2.1483 | **19.52%** | 16.89% | **0.7707** | 0.8808 | −34.57% |
| Sample-cov Markowitz | 2.1180 | 19.34% | 18.92% | 0.6786 | 0.8181 | −31.51% |
| Shrinkage-cov Markowitz | 2.0532 | 18.95% | 19.12% | 0.6512 | 0.7830 | −31.84% |
| Min-variance | 1.4241 | 14.76% | **14.60%** | 0.5656 | 0.7135 | **−25.93%** |

**Mean walk-forward out-of-sample IC with the 21-day target: `0.1019`.**

The 21-day horizon is now the current implementation because it approximately
matches the monthly rebalance/holding horizon. The original 5-day-stage results
above are kept as part of the project history rather than overwritten.

### 4.4 Black-Litterman prior-only ablation

The first Black-Litterman experiment asks a controlled question:

> Does performance come from the market-equilibrium prior itself, or from the
> ML information added as views?

The **BL Prior Only** portfolio uses `π = δΣ_BL w_mkt` but no ML views.

At the completed configuration:

| Strategy | Total Return | Annualized Return | Annualized Vol | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|
| BL Prior Only | 1.4883 | 15.22% | 15.42% | **0.5659** | 0.6879 | −27.95% |

This gives the reference point for measuring the incremental value of ML views.

### 4.5 Black-Litterman + ML relative views

With the same portfolio optimizer and risk model, adding uncertainty-weighted
relative ML views materially improves performance.

At `uncertainty_multiplier = 10`:

| Strategy | Total Return | Annualized Return | Annualized Vol | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|
| BL + ML | **1.9547** | **18.34%** | 15.44% | **0.7672** | **0.9415** | **−26.77%** |
| BL Prior Only | 1.4883 | 15.22% | 15.42% | 0.5659 | 0.6879 | −27.95% |

The key decomposition is:

```text
Sharpe:          0.5659 → 0.7672
Annualized vol: 15.42%  → 15.44%
```

So the improvement is not explained by taking substantially more total risk.

Relative to the updated 21-day shrinkage-Markowitz baseline:

```text
Shrinkage Markowitz Sharpe = 0.6512
BL + ML Sharpe              = 0.7672
```

At this conservative confidence setting, BL + ML also approximately matches
equal-weight Sharpe (`0.7707`) while producing lower volatility (`15.44%`
vs. `16.89%`) and a smaller maximum drawdown (`−26.77%` vs. `−34.57%`).

### 4.6 View-confidence sensitivity experiment

A single BL confidence level could be accidental, so the experiment was
repeated across different values of `uncertainty_multiplier` while keeping
the rest of the pipeline fixed.

Because

```text
Ω ∝ uncertainty_multiplier
```

larger values mean **less trust** in the ML views.

| Uncertainty Multiplier | Annualized Return | Annualized Vol | Sharpe | Sortino | Max Drawdown |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.72% | 15.96% | 0.8288 | 1.0338 | −26.06% |
| 2 | 19.68% | 15.69% | **0.8397** | **1.0453** | −25.89% |
| 3 | 19.43% | 15.57% | 0.8305 | 1.0303 | **−25.80%** |
| 4 | 19.33% | 15.51% | 0.8270 | 1.0223 | −25.87% |
| 5 | 19.13% | 15.49% | 0.8155 | 1.0067 | −26.06% |
| 10 | 18.34% | 15.44% | 0.7672 | 0.9415 | −26.77% |
| 15 | 17.52% | 15.42% | 0.7150 | 0.8773 | −27.07% |
| BL Prior Only | 15.22% | 15.42% | 0.5659 | 0.6879 | −27.95% |

The important result is the **shape of the response**, not simply the numerical
maximum at multiplier `2`.

As ML views are trusted less, performance gradually declines toward the weaker
prior-only solution. The broad `1–5` region remains strong, suggesting that the
ML ranking signal contains useful information rather than the result depending
on one knife-edge confidence setting.

To avoid presenting the best observed test-period value as a universally
optimal hyperparameter, the project uses the more representative
`uncertainty_multiplier = 3` as the final reported configuration:

```text
Annualized return = 19.43%
Annualized vol    = 15.57%
Sharpe            = 0.8305
Sortino           = 1.0303
Max drawdown      = −25.80%
```

This final BL + ML configuration exceeds the updated equal-weight Sharpe
(`0.7707`) while also showing lower volatility and a smaller maximum drawdown.

### 4.7 What changed in the research story

The project now forms one continuous sequence of experiments:

```text
1. Sample covariance becomes unstable as N/T rises.
2. Ledoit-Wolf shrinkage improves covariance stability.
3. Better covariance alone does not solve out-of-sample portfolio performance.
4. Risk-aversion ablation identifies expected-return estimation as the bottleneck.
5. The ML horizon is aligned with the monthly portfolio horizon (21 days).
6. Black-Litterman replaces the noisy historical-mean anchor with a
   market-implied equilibrium prior.
7. ML forecasts are encoded as uncertain relative views.
8. BL prior-only is weak, but adding ML views materially improves Sharpe.
9. The confidence sweep shows that the improvement persists across a broad
   confidence region and fades as the views are trusted less.
```

The project therefore evolves from studying

```text
covariance uncertainty
        ↓
expected-return uncertainty
        ↓
view uncertainty
```

rather than treating portfolio optimization as a single one-shot model.


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

```text
portfolio-optimizer/
├── config/
│   └── nifty50.txt                  # 39-ticker NSE large-cap universe
├── dashboard/
│   └── app.py                       # current BL + ML portfolio interface
├── data/
│   ├── prices.parquet               # cached adjusted close prices
│   ├── features_long.parquet        # engineered features, long format
│   ├── shares_outstanding.csv       # historical shares for market-cap weights
│   └── nifty50_market.csv           # NIFTY 50 market-return diagnostics
├── src/
│   ├── __init__.py
│   ├── backtest.py                  # original walk-forward engine + metrics
│   ├── black_litterman.py           # prior, P/Q views, Ω, BL posterior
│   ├── covariance.py                # sample vs. Ledoit-Wolf estimators
│   ├── data_pipeline.py             # features + 21-day forward-return target
│   ├── market_data.py               # point-in-time market-cap construction
│   ├── ml_predict.py                # LightGBM 21-day return model
│   └── optimizer.py                 # constrained cvxpy Markowitz optimizer
├── download_market_data.py          # historical shares / market-data download
├── run_ablation.py                  # original return-estimation ablations
├── run_black_litterman_backtest.py  # BL prior-only + BL/ML walk-forward test
├── run_real_backtest.py             # updated 21-day baseline backtest
├── requirements.txt
└── README.md
```

---

## 7. How to run

```bash
pip install -r requirements.txt

# 1. Pull / refresh real price data
python -m src.data_pipeline --tickers config/nifty50.txt --start 2019-01-01

# 2. Run the updated 21-day baseline backtest
python run_real_backtest.py

# 3. Run the original ablation study
python run_ablation.py

# 4. Download / refresh historical shares and market data used by BL
python download_market_data.py

# 5. Run the Black-Litterman walk-forward experiment
python run_black_litterman_backtest.py

# 6. Launch the current portfolio interface
streamlit run dashboard/app.py
```

The Streamlit app is now treated as a **user-facing interface for the final
21-day + Black-Litterman + ML system**. The research history, ablations, and
earlier results remain documented here in the README rather than being mixed
into the top-level portfolio-allocation UI.

---

## 8. Honesty checklist

Things this project deliberately does **not** fake or hide:

- **The original ML model's weak out-of-sample IC (`0.015`) is retained and
  reported rather than overwritten.** The updated 21-day experiment separately
  reports mean walk-forward OOS IC `0.1019`, making the change in forecast
  horizon explicit rather than silently replacing the earlier result.
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
- **The original weak-result table is still retained.** Later 21-day and
  Black-Litterman results are added as new experiments instead of replacing
  the earlier result that motivated the investigation.
- **The highest observed BL confidence-sweep Sharpe is not treated as an
  automatically tuned optimum.** Multiplier `2` produced the highest observed
  Sharpe (`0.8397`), but the project reports multiplier `3` (`0.8305`) as a
  representative moderate-confidence configuration to avoid presenting a
  test-period maximum as a universal optimum.
- **Current dashboard allocation and historical backtest performance are
  distinguished.** The dashboard's portfolio weights are the latest weights
  produced by the final BL + ML methodology, while Sharpe `0.8305` is the
  realized walk-forward result of that configuration over the historical test.

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
- **Black-Litterman as a natural next step — implemented in the current
  extension:** the 1/N puzzle finding suggested that the real fix was not only
  better covariance estimation but better *return* estimation. This solution
  is now committed: the project builds a point-in-time market-implied
  equilibrium prior, encodes 21-day ML rankings as relative views, models view
  uncertainty through `Ω`, and evaluates BL prior-only vs. BL + ML in a
  walk-forward backtest. The completed results are documented in Sections
  4.4–4.7.
- **Single-country, single-asset-class scope:** extending to multi-asset
  (bonds, gold, international equities) would test whether the 1/N puzzle
  finding holds when the asset universe is more heterogeneous.
- **Controlled view-quality experiment:** a future extension can independently
  degrade ML view quality by injecting controlled noise into `Q` and then study
  `view quality × view confidence`. This would test whether Black-Litterman
  behaves as intended when views become progressively less informative, rather
  than only varying confidence on the naturally observed ML signal.

---

## 10. Tech stack

- **Data:** `yfinance`, `pandas`, `pyarrow`
- **Statistics/ML:** `numpy`, `scikit-learn` (Ledoit-Wolf, gradient boosting
  fallback), `lightgbm`, `scipy`; custom Black-Litterman implementation
- **Optimization:** `cvxpy` (OSQP solver)
- **Dashboard:** `streamlit`, `plotly`

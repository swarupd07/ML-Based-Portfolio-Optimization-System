# ML-Based Portfolio Optimization System

An end-to-end quantitative finance project combining **LightGBM return prediction**, **Ledoit-Wolf covariance shrinkage**, **Black-Litterman expected returns**, and **constrained Markowitz optimization**, evaluated using a strict walk-forward backtest with transaction costs and turnover limits.

The main research question evolved from:

> Does better covariance estimation improve portfolio performance?

into:

> If covariance estimation is stabilized, is expected-return estimation still the main bottleneck?

---

## 1. System Overview

```text
NSE large-cap prices
        ↓
Feature engineering
(momentum, volatility, RSI, MACD)
        ↓
Leakage-safe 21-day LightGBM forecasts
        ↓
Relative ML views
        ↓
Black-Litterman posterior
(market-implied prior + uncertain ML views)
        ↓
Ledoit-Wolf risk covariance
        ↓
Constrained Markowitz optimizer
        ↓
Monthly walk-forward backtest
(10 bps costs + turnover cap)
```

**Universe:** 39 NSE large-cap stocks  
**Period:** 2019-01-01 to 2026-08-20  
**Rebalance frequency:** monthly  
**Maximum stock weight:** 15%  
**Turnover cap:** 0.5 L1 per rebalance  
**Transaction cost:** 10 bps per unit turnover  

---

## 2. Methodology

### 2.1 Features and ML target

Features:

- `mom_5d`, `mom_21d`
- `vol_21d`, `vol_63d`
- `ma_ratio_10_50`
- `rsi_14`
- `macd_hist`

Current ML target:

```text
fwd_return_21d
```

The 21-day horizon approximately matches the monthly holding period.

### Leakage-safe training

A 21-day forward return attached to date `t` requires the price at `t+21`.

Therefore, training only on:

```text
date < rebalance_date
```

is not sufficient.

The corrected pipeline **purges the final 21 trading dates before each rebalance**, ensuring every training label is fully observable before the decision date.

Corrected mean walk-forward OOS Information Coefficient:

```text
IC = 0.0225
```

---

## 3. Covariance Estimation

The project compares sample covariance with Ledoit-Wolf shrinkage.

```text
Σ_shrunk = δF + (1 − δ)Σ_sample
```

Shrinkage becomes more useful as the estimation window becomes shorter:

| Window | Sample Condition # | LW Condition # | Improvement |
|---|---:|---:|---:|
| ~1880 days | 60.5 | 55.0 | ~9% |
| 756 days | 55.2 | 47.1 | ~15% |
| 378 days | 69.9 | 55.3 | ~21% |
| 189 days | 109.5 | 64.7 | **~41%** |

**Finding:** Ledoit-Wolf improves covariance stability, especially when `N/T` becomes less favorable.

---

## 4. Original Experiment: the 1/N Puzzle

The original project used historical mean returns with a small ML tilt.

Historical result:

| Strategy | Sharpe |
|---|---:|
| Equal Weight | **0.79** |
| Sample-Cov Markowitz | 0.28 |
| Shrinkage-Cov Markowitz | 0.26 |
| Minimum Variance | 0.60 |

Original ML OOS IC was approximately:

```text
0.015
```

The surprising result was that **equal-weight outperformed all optimized portfolios**.

A risk-aversion sweep then showed:

| Risk Aversion | Sharpe |
|---:|---:|
| 3 | 0.186 |
| 8 | 0.435 |
| 15 | 0.569 |
| 25 | 0.594 |
| Minimum Variance | **0.596** |

As reliance on expected returns decreased, performance approached the minimum-variance solution.

**Conclusion:** covariance estimation was not the dominant problem; **expected-return estimation was**.

---

## 5. Corrected 21-Day Baselines

After changing the ML horizon to 21 days **and fixing forward-label leakage**, the corrected baseline results are:

| Strategy | Annual Return | Volatility | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| Equal Weight | **19.52%** | 16.89% | **0.7707** | 0.8808 | −34.57% |
| Sample-Cov Markowitz | 14.41% | 19.11% | 0.4140 | 0.5041 | −33.87% |
| Shrinkage-Cov Markowitz | 14.16% | 19.25% | 0.3977 | 0.4824 | −33.83% |
| Minimum Variance | 14.76% | **14.60%** | 0.5656 | 0.7135 | **−25.93%** |

The corrected ML IC is only **0.0225**, which is small but more realistic for cross-sectional equity prediction.

---

## 6. Black-Litterman Extension

The BL equilibrium prior is:

```text
π = δ Σ_BL w_mkt
```

where:

- `δ = 2.5`
- `w_mkt` = historically dated market-cap weights
- `Σ_BL` = Ledoit-Wolf covariance using up to 756 trading days

The portfolio-risk covariance uses the most recent 252 trading days.

### Relative ML views

The ML model is used as a ranking signal rather than as an absolute-return oracle.

```text
winner − loser = predicted relative outperformance
```

Each view is represented through `P` and `Q`.

The 21-day predicted spread is annualized using:

```text
252 / 21
```

and capped at `±15%`.

View uncertainty is:

```text
Ω = diag(P (τΣ) P') × uncertainty_multiplier
```

with:

```text
τ = 0.05
n_views = 10
```

---

## 7. BL Results

### BL prior-only

| Strategy | Annual Return | Volatility | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| BL Prior Only | 15.22% | 15.42% | **0.5659** | −27.95% |

### Leakage-safe confidence sweep

| Uncertainty Multiplier | Annual Return | Volatility | Sharpe | Max Drawdown |
|---:|---:|---:|---:|---:|
| **1** | **17.12%** | 16.18% | **0.6563** | −27.83% |
| 2 | 16.44% | 15.98% | 0.6221 | −27.47% |
| 3 | 15.84% | 15.87% | 0.5881 | −27.44% |
| 5 | 15.74% | 15.76% | 0.5859 | **−27.34%** |
| 10 | 15.64% | 15.60% | 0.5861 | −27.40% |
| 15 | 15.47% | 15.52% | 0.5777 | −27.57% |
| BL Prior Only | 15.22% | 15.42% | 0.5659 | −27.95% |

The strongest observed tested setting is:

```text
uncertainty_multiplier = 1
Sharpe                 = 0.6563
Annual return          = 17.12%
Annual volatility      = 16.18%
Sortino                = 0.7927
Max drawdown           = −27.83%
```

This is treated as the **best observed setting in the completed sweep**, not as a universally optimal confidence value.

### Main finding

As view uncertainty increases, BL + ML performance moves toward the weaker prior-only portfolio.

This suggests the ML ranking signal contains some useful information, but the improvement is **modest rather than dramatic**:

```text
BL Prior Only Sharpe = 0.5659
BL + ML Sharpe       = 0.6563
Equal Weight Sharpe  = 0.7707
```

So Black-Litterman partially improves the return-estimation problem, but **equal-weight still remains the strongest Sharpe baseline**.

---

## 8. Research Progression

```text
Covariance instability
        ↓
Ledoit-Wolf shrinkage
        ↓
Better Σ but weak portfolio performance
        ↓
Expected-return estimation identified as bottleneck
        ↓
21-day ML forecast aligned to holding horizon
        ↓
Forward-label leakage identified and removed
        ↓
Black-Litterman equilibrium prior
        ↓
ML encoded as uncertain relative views
        ↓
Confidence sensitivity experiment
```

The project therefore studies three related sources of uncertainty:

```text
covariance uncertainty
        ↓
expected-return uncertainty
        ↓
view uncertainty
```

---

## 9. Project Structure

```text
portfolio-optimizer/
├── config/
│   └── nifty50.txt
├── dashboard/
│   └── app.py
├── data/
│   ├── prices.parquet
│   ├── features_long.parquet
│   ├── shares_outstanding.csv
│   └── nifty50_market.csv
├── src/
│   ├── __init__.py
│   ├── backtest.py
│   ├── black_litterman.py
│   ├── covariance.py
│   ├── data_pipeline.py
│   ├── market_data.py
│   ├── ml_predict.py
│   └── optimizer.py
├── download_market_data.py
├── run_ablation.py
├── run_black_litterman_backtest.py
├── run_real_backtest.py
├── requirements.txt
└── README.md
```

---

## 10. How to Run

```bash
pip install -r requirements.txt

python -m src.data_pipeline --tickers config/nifty50.txt --start 2019-01-01

python run_real_backtest.py

python run_ablation.py

python download_market_data.py

python run_black_litterman_backtest.py --uncertainty-multiplier 1

streamlit run dashboard/app.py
```

---

## 11. Limitations and Future Work

- **Survivorship bias:** today's 39-stock universe is used historically.
- **No direct NIFTY 50 benchmark** is included yet.
- **Single-country / single-asset-class universe.**
- **Black-Litterman extension — completed:** market-implied prior, relative ML views, and confidence sensitivity are now implemented.
- **Future:** inject controlled noise into `Q` and study `view quality × view confidence`.
- Historical shares outstanding are aligned as-of each rebalance date, but Yahoo data is not equivalent to an institutional point-in-time fundamentals database.

---

## 12. Honesty / Validity Notes

- Original weak results are retained instead of overwritten.
- The 21-day target initially produced inflated IC and Sharpe because the training boundary did not fully account for the forward-label horizon.
- The final pipeline explicitly **purges the last 21 trading dates before each rebalance**.
- After the fix, OOS IC fell from the inflated value to **0.0225**, and BL + ML Sharpe fell to a more realistic **0.6563** at the strongest tested confidence.
- Equal-weight still wins on Sharpe, and this is reported directly.
- All backtests include transaction costs and turnover controls.

---

## 13. Tech Stack

- **Data:** `yfinance`, `pandas`, `pyarrow`
- **ML / statistics:** `numpy`, `scikit-learn`, `lightgbm`, `scipy`
- **Risk estimation:** Ledoit-Wolf shrinkage
- **Expected returns:** custom Black-Litterman implementation
- **Optimization:** `cvxpy`
- **Dashboard:** `streamlit`, `plotly`

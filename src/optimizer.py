# Markowitz mean-variance optimizer, solved with cvxpy

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------

def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 3.0,
    max_weight: float = 0.20,
    min_weight: float = 0.0,
    long_only: bool = True,
    prev_weights: pd.Series | None = None,
    turnover_limit: float | None = None,
) -> pd.Series:
    """Solve: maximize  w'mu - risk_aversion * w'Sigma w
              s.t.       sum(w) == 1

    Parameters
    ----------
    1. expected_returns : annualized expected return per asset (from ML model or historical mean)
    2. cov_matrix : annualized covariance matrix (sample or shrinkage estimate).
    3. risk_aversion : higher = more conservative. 0-10 aggressive, 11-20 balanced, 20+ conservative.
    4. max_weight : cap on any single position
    5. prev_weights, turnover_limit : if given, constrains how much the new allocation can differ from the previous one.(keep transaction costs bounded during rebalancing)

    Returns
    -------
    pd.Series of weights indexed the same as expected_returns (summing to 1)
    """

    tickers = expected_returns.index
    n = len(tickers)
    mu = expected_returns.reindex(tickers).values
    Sigma = cov_matrix.reindex(index=tickers, columns=tickers).values
    # Ensuring symmetric PSD (numerical safety net for optimizer stability)
    Sigma = (Sigma + Sigma.T) / 2

    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, cp.psd_wrap(Sigma)))

    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= min_weight)
    constraints.append(w <= max_weight)

    if prev_weights is not None and turnover_limit is not None:
        w_prev = prev_weights.reindex(tickers).fillna(0).values
        constraints.append(cp.norm1(w - w_prev) <= turnover_limit)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimizer failed to converge: status={problem.status}")

    weights = pd.Series(np.clip(w.value, 0, None), index=tickers)
    weights = weights / weights.sum()  # renormalize to kill tiny numerical drift
    return weights

# ------------------------------------------------------------------------------------

def min_variance_portfolio(cov_matrix: pd.DataFrame, max_weight: float = 0.20) -> pd.Series:
    # The pure minimum-variance portfolio (ignores expected returns entirely)
    tickers = cov_matrix.index
    n = len(tickers)
    Sigma = (cov_matrix.values + cov_matrix.values.T) / 2

    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    weights = pd.Series(np.clip(w.value, 0, None), index=tickers)
    return weights / weights.sum()

# ------------------------------------------------------------------------------------

def portfolio_stats(weights: pd.Series, expected_returns: pd.Series, cov_matrix: pd.DataFrame,risk_free_rate: float = 0.065) -> dict:
    # Expected return, volatility, and Sharpe ratio for a given weight vector.
    # risk_free_rate defaults to a rough Indian 10Y G-Sec yield ( 0.065 = 6.5% )
    w = weights.reindex(expected_returns.index).values
    mu = expected_returns.values
    Sigma = cov_matrix.reindex(index=expected_returns.index, columns=expected_returns.index).values

    ret = float(w @ mu)
    vol = float(np.sqrt(w @ Sigma @ w))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
    return {"expected_return": ret, "volatility": vol, "sharpe_ratio": sharpe}

# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_pipeline import synthetic_prices
    from src.covariance import shrinkage_covariance

    prices = synthetic_prices()
    rets = prices.pct_change().dropna()
    mu = rets.mean() * 252  # naive historical mean expected return, just for a test
    cov, delta = shrinkage_covariance(rets)

    w = optimize_portfolio(mu, cov, risk_aversion=3.0, max_weight=0.15)
    print(w.sort_values(ascending=False).head(10))
    print(portfolio_stats(w, mu, cov))

# ------------------------------------------------------------------------------------
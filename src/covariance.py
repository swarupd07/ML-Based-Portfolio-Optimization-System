# Covariance estimation: naive sample covariance vs. Ledoit-Wolf shrinkage.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

# ------------------------------------------------------------------------------------

def sample_covariance(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    #Naive sample covariance matrix
    cov = returns.cov()
    if annualize:
        cov = cov * 252
    return cov

# ------------------------------------------------------------------------------------

def shrinkage_covariance(returns: pd.DataFrame, annualize: bool = True) -> tuple[pd.DataFrame, float]:
    # Ledoit-Wolf shrinkage covariance matrix
    lw = LedoitWolf().fit(returns.dropna().values)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    if annualize:
        cov = cov * 252
    return cov, lw.shrinkage_

# ------------------------------------------------------------------------------------

def condition_number(cov: pd.DataFrame) -> float:
    # Ratio of largest to smallest eigenvalue
    eigvals = np.linalg.eigvalsh(cov.values)
    eigvals = eigvals[eigvals > 1e-12]
    return float(eigvals.max() / eigvals.min())

# ------------------------------------------------------------------------------------

def compare_estimators(returns: pd.DataFrame) -> pd.DataFrame:
    # Convenience function: side-by-side comparison table used in the dashboard to justify the shrinkage choice
    samp = sample_covariance(returns)
    shrunk, delta = shrinkage_covariance(returns)

    rows = {
        "condition_number": [condition_number(samp), condition_number(shrunk)],
        "mean_abs_correlation_off_diag": [
            _mean_abs_offdiag_corr(samp),
            _mean_abs_offdiag_corr(shrunk),
        ],
        "frobenius_norm": [
            np.linalg.norm(samp.values, "fro"),
            np.linalg.norm(shrunk.values, "fro"),
        ],
    }
    df = pd.DataFrame(rows, index=["sample", f"ledoit_wolf (delta={delta:.3f})"]).T
    return df

# ------------------------------------------------------------------------------------

def _mean_abs_offdiag_corr(cov: pd.DataFrame) -> float:
    std = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(std, std)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.abs(corr[mask]).mean())

# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_pipeline import synthetic_prices

    prices = synthetic_prices()
    rets = prices.pct_change().dropna()
    print(compare_estimators(rets))

# ------------------------------------------------------------------------------------
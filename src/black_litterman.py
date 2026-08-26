import pandas as pd
import numpy as np


# Prior: π=δΣw(m) 
def implied_equilibrium_returns(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float) -> pd.Series:

    tickers = cov_matrix.index

    w = market_weights.reindex(tickers).fillna(0.0)

    if w.sum() <= 0:
        raise ValueError("Market weights must have positive total weight.")

    w = w / w.sum()

    pi = risk_aversion * cov_matrix @ w

    return pd.Series(pi, index=tickers, name="bl_prior")


def normalize_market_weights(market_caps: pd.Series,tickers: pd.Index,) -> pd.Series:

    caps = market_caps.reindex(tickers).fillna(0.0)
    if caps.sum() <= 0:
        raise ValueError("Market caps must have positive total value.")

    return caps / caps.sum()

def estimate_market_risk_aversion( market_returns: pd.Series, risk_free_rate: float = 0.065,) -> float:

    ann_return = market_returns.mean() * 252
    ann_var = market_returns.var() * 252

    if ann_var <= 0:
        raise ValueError("Market variance must be positive.")

    delta = (ann_return - risk_free_rate) / ann_var

    return float(delta)



# Building P, Q, and Ω 

def build_relative_views(predictions: pd.Series, tickers: pd.Index, n_views: int = 10):
    preds = predictions.reindex(tickers).dropna()

    ranked = preds.sort_values()

    n_views = min(n_views, len(ranked) // 2)

    losers = ranked.index[:n_views]
    winners = ranked.index[-n_views:][::-1]

    P = np.zeros((n_views, len(tickers)))
    Q = np.zeros(n_views)

    ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}

    for k, (winner, loser) in enumerate(zip(winners, losers)):
        P[k, ticker_to_idx[winner]] = 1.0
        P[k, ticker_to_idx[loser]] = -1.0

        annualized_view = (predictions[winner] - predictions[loser]) * (252 / 21) # annualizing returns as my ml model gives 21-day return and without annualization it will be 21 day ralative performance
        Q[k] = np.clip(annualized_view,-0.15,0.15)        # even if the ML model predicts a huge 21-day spread, we do not allow a single relative view to claim more than 15% annual outperformance.

    return P, Q

# Ω = diag(P(τΣ)P.T)

def build_omega(P: np.ndarray,cov_matrix: pd.DataFrame,tau: float, uncertainty_multiplier: float = 10.0) -> np.ndarray:

    Sigma = cov_matrix.values

    view_cov = P @ (tau * Sigma) @ P.T

    Omega = np.diag(np.diag(view_cov))* uncertainty_multiplier # assuming the views are independent

    return Omega

# A = (τΣ)^{−1}
# B = P.TΩ^{−1}
# μBL ​= [A.π + B.Q] / [A + B.P]      => prior + views, weighted by uncertainty

def black_litterman_posterior(
    prior: pd.Series,
    cov_matrix: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05 ) -> pd.Series:

    tickers = cov_matrix.index

    pi = prior.reindex(tickers).values
    Sigma = cov_matrix.values

    tau_sigma_inv = np.linalg.inv(tau * Sigma)
    omega_inv = np.linalg.inv(Omega)

    posterior_cov_inv = (tau_sigma_inv + P.T @ omega_inv @ P )

    posterior_mean_term = ( tau_sigma_inv @ pi + P.T @ omega_inv @ Q )

    mu_bl = np.linalg.solve( posterior_cov_inv, posterior_mean_term) # more stable that np.linalg.inv(posterior_cov_inv) @ posterior_mean_term

    return pd.Series( mu_bl, index=tickers, name="bl_expected_return", )


def build_bl_expected_returns(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float,
    ml_predictions: pd.Series,
    tau: float = 0.05,
    n_views: int = 10,
    uncertainty_multiplier: float = 10.0, ) -> pd.Series:

    tickers = cov_matrix.index

    # 1. Normalizing market weights
    w_mkt = market_weights.reindex(tickers).fillna(0.0)
    w_mkt = w_mkt / w_mkt.sum()

    # 2. Equilibrium prior
    prior = implied_equilibrium_returns( cov_matrix=cov_matrix, market_weights=w_mkt, risk_aversion=risk_aversion )

    # 3. ML-based relative views
    P, Q = build_relative_views( predictions=ml_predictions, tickers=tickers, n_views=n_views)

    # 4. View uncertainty
    Omega = build_omega( P=P, cov_matrix=cov_matrix, tau=tau, uncertainty_multiplier=uncertainty_multiplier )

    # 5. Black-Litterman posterior
    mu_bl = black_litterman_posterior(
        prior=prior,
        cov_matrix=cov_matrix,
        P=P,
        Q=Q,
        Omega=Omega,
        tau=tau,
    )

    return mu_bl


if __name__ == "__main__":
    tickers = pd.Index(["A", "B", "C", "D"])

    cov_matrix = pd.DataFrame(
        [
            [0.040, 0.010, 0.008, 0.006],
            [0.010, 0.050, 0.009, 0.007],
            [0.008, 0.009, 0.030, 0.005],
            [0.006, 0.007, 0.005, 0.045],
        ],
        index=tickers,
        columns=tickers,
    )

    market_weights = pd.Series(
        [0.40, 0.30, 0.20, 0.10],
        index=tickers,
    )

    ml_predictions = pd.Series(
        [0.015, -0.005, 0.010, -0.010],
        index=tickers,
    )

    prior = implied_equilibrium_returns(
    cov_matrix=cov_matrix,
    market_weights=market_weights,
    risk_aversion=2.5)

    mu_bl = build_bl_expected_returns(
        cov_matrix=cov_matrix,
        market_weights=market_weights,
        risk_aversion=2.5,
        ml_predictions=ml_predictions,
        tau=0.05,
        n_views=2,
    )

    print("\nPrior:")
    print(prior)

    print("\nBL Posterior:")
    print(mu_bl)

    print("\nChange:")
    print(mu_bl - prior)
    print(mu_bl)
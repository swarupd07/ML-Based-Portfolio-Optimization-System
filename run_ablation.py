import pandas as pd
from src.data_pipeline import engineer_features, build_long_dataset
from src.covariance import sample_covariance, shrinkage_covariance
from src.optimizer import optimize_portfolio, portfolio_stats, min_variance_portfolio
from src.ml_predict import historical_mean_baseline
from src.backtest import BacktestConfig, _rebalance_dates, performance_metrics

prices = pd.read_parquet("data/prices.parquet")
feats = engineer_features(prices)
long_df = build_long_dataset(feats)

daily_returns = prices.pct_change()
cfg = BacktestConfig()
rebal_dates = _rebalance_dates(prices.index, cfg.rebalance_freq, cfg.lookback_min_days)

prev_w = None
rets = None

for i, d in enumerate(rebal_dates):
    window = daily_returns[daily_returns.index < d].tail(cfg.lookback_min_days).dropna(axis=1)
    tickers = window.columns
    mu = historical_mean_baseline(long_df, train_end=d).reindex(tickers).fillna(0)  # NO ml tilt
    cov, _ = shrinkage_covariance(window)
    w = optimize_portfolio(mu, cov, risk_aversion=cfg.risk_aversion, max_weight=cfg.max_weight,
                            prev_weights=prev_w, turnover_limit=cfg.turnover_limit)
    end = rebal_dates[i+1] if i+1 < len(rebal_dates) else prices.index[-1]
    period = daily_returns.loc[d:end, tickers].iloc[1:]
    turnover = (w - prev_w).abs().sum() if prev_w is not None else w.abs().sum()
    g = period @ w
    if len(g) > 0:
        g.iloc[0] -= turnover * cfg.cost_bps / 10000
    rets = pd.concat([rets, g]) if rets is not None else g
    prev_w = w

print(performance_metrics(rets))
import pandas as pd
from src.data_pipeline import engineer_features, build_long_dataset
from src.backtest import run_backtest, BacktestConfig

prices = pd.read_parquet("data/prices.parquet")
feats = engineer_features(prices)
long_df = build_long_dataset(feats)

cfg = BacktestConfig()  # monthly rebalance, 10bps cost, 15% max weight, 252-day lookback
results, metrics, ic_df = run_backtest(prices, long_df, cfg)

print("=== Performance ===")
print(metrics.round(4))
print("\n=== Mean out-of-sample IC ===")
print(ic_df["ic"].mean())
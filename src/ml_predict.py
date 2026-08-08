# ML expected-return prediction

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEATURE_COLS = [
    "mom_5d", "mom_21d", "vol_21d", "vol_63d",
    "ma_ratio_10_50", "rsi_14", "macd_hist",
]
TARGET_COL = "fwd_return_5d"

# ------------------------------------------------------------------------------------

def _get_model():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=4,
            num_leaves=15, min_child_samples=30, subsample=0.8,
            colsample_bytree=0.8, verbosity=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.03, max_iter=200)

# ------------------------------------------------------------------------------------

def train_predict(
    long_df: pd.DataFrame,
    train_end: pd.Timestamp,
    predict_date: pd.Timestamp,
) -> tuple[pd.Series, dict]:
    """Train on all data strictly before `train_end`, predict expected forward
    returns as of `predict_date`sees past data.

    Returns
    -------
    predictions : pd.Series indexed by ticker, the predicted fwd_return_5d
    diagnostics : dict with n_train_rows, in-fold IC, etc. for logging
    """
    train = long_df[long_df["date"] < train_end].dropna(subset=FEATURE_COLS + [TARGET_COL])

    if len(train) < 200:
        # not enough history yet so fall back to zero signal (equal weight)
        tickers = long_df["ticker"].unique()
        return pd.Series(0.0, index=tickers), {"n_train_rows": len(train), "status": "insufficient_data"}

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]

    model = _get_model()
    model.fit(X_train, y_train)

    predict_rows = long_df[long_df["date"] == predict_date].dropna(subset=FEATURE_COLS)
    if predict_rows.empty:
        tickers = long_df["ticker"].unique()
        return pd.Series(0.0, index=tickers), {"n_train_rows": len(train), "status": "no_predict_rows"}

    preds = model.predict(predict_rows[FEATURE_COLS])
    pred_series = pd.Series(preds, index=predict_rows["ticker"].values)

    train_preds = model.predict(X_train)
    ic, _ = spearmanr(train_preds, y_train)

    diagnostics = {
        "n_train_rows": len(train),
        "n_predicted": len(pred_series),
        "train_ic": ic,
        "status": "ok",
    }
    return pred_series, diagnostics

# ------------------------------------------------------------------------------------

def historical_mean_baseline(long_df: pd.DataFrame, train_end: pd.Timestamp) -> pd.Series:
    # The naive baseline expected-return estimate
    train = long_df[long_df["date"] < train_end]
    mean_ret = train.groupby("ticker")["returns"].mean() * 252
    return mean_ret

# ------------------------------------------------------------------------------------

def out_of_sample_ic(predictions: pd.Series, actuals: pd.Series) -> float:
    # Spearman rank correlation between predicted and realized forward returns
    joined = pd.concat([predictions.rename("pred"), actuals.rename("actual")], axis=1).dropna()
    if len(joined) < 5:
        return np.nan
    ic, _ = spearmanr(joined["pred"], joined["actual"])
    return ic

# ------------------------------------------------------------------------------------
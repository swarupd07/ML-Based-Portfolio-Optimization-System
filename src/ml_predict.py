# ML expected-return prediction

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEATURE_COLS = [
    "mom_5d",
    "mom_21d",
    "vol_21d",
    "vol_63d",
    "ma_ratio_10_50",
    "rsi_14",
    "macd_hist",
]

TARGET_COL = "fwd_return_21d"
TARGET_HORIZON_DAYS = 21
RANDOM_STATE = 42


# ------------------------------------------------------------------------------------
def _get_model():
    """
    Return the primary LightGBM model, with a sklearn fallback.

    A fixed random_state is used so repeated runs are reproducible.
    """
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
            random_state=RANDOM_STATE,
        )

    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.03,
            max_iter=200,
            random_state=RANDOM_STATE,
        )


# ------------------------------------------------------------------------------------
def _purged_training_cutoff(
    long_df: pd.DataFrame,
    train_end: pd.Timestamp,
    horizon_days: int = TARGET_HORIZON_DAYS,
) -> pd.Timestamp:
    """
    Return the latest feature date whose forward target is fully observable
    before `train_end`.

    Why this is necessary
    ---------------------
    fwd_return_21d attached to feature date t uses the price at t+21.

    Therefore simply training on:

        date < train_end

    is NOT enough. Rows close to train_end would have labels that use prices
    from after the rebalance date.

    We conservatively remove the final `horizon_days` trading dates before
    train_end. This ensures every training label is based only on information
    available strictly before the decision date.

    Example
    -------
    If train_end is 2024-06-03, rows from roughly the previous 21 trading
    sessions are excluded from training because their 21-day forward returns
    are not yet fully known.
    """
    train_end = pd.Timestamp(train_end)

    available_dates = pd.DatetimeIndex(
        pd.to_datetime(long_df.loc[long_df["date"] < train_end, "date"].unique())
    ).sort_values()

    if len(available_dates) <= horizon_days:
        raise ValueError(
            f"Not enough dates before {train_end.date()} to apply a "
            f"{horizon_days}-trading-day purge."
        )

    # Remove the final horizon_days feature dates.
    return pd.Timestamp(available_dates[-(horizon_days + 1)])


# ------------------------------------------------------------------------------------
def train_predict(
    long_df: pd.DataFrame,
    train_end: pd.Timestamp,
    predict_date: pd.Timestamp,
    target_horizon_days: int = TARGET_HORIZON_DAYS,
) -> tuple[pd.Series, dict]:
    """
    Train the ML return model without forward-label leakage.

    Parameters
    ----------
    long_df:
        Long-format feature table with columns:
        date, ticker, FEATURE_COLS, and TARGET_COL.

    train_end:
        Portfolio decision / rebalance date.

    predict_date:
        Date whose features are used to generate the cross-sectional forecast.

    target_horizon_days:
        Forward target horizon in trading days. Current project target = 21.

    Leakage rule
    ------------
    The target at feature date t is fwd_return_21d, which uses price t+21.
    Therefore the final 21 trading dates before train_end are purged from the
    training sample.

    This is stricter than merely requiring date < train_end and prevents
    forward-return labels from crossing the rebalance boundary.

    Returns
    -------
    predictions:
        pd.Series indexed by ticker containing predicted forward 21-day returns.

    diagnostics:
        Metadata including number of training rows, purge cutoff, and in-sample IC.
    """
    train_end = pd.Timestamp(train_end)
    predict_date = pd.Timestamp(predict_date)

    cutoff_date = _purged_training_cutoff(
        long_df=long_df,
        train_end=train_end,
        horizon_days=target_horizon_days,
    )

    train = long_df[
        long_df["date"] <= cutoff_date
    ].dropna(
        subset=FEATURE_COLS + [TARGET_COL]
    )

    if len(train) < 200:
        tickers = long_df["ticker"].unique()

        return (
            pd.Series(0.0, index=tickers),
            {
                "n_train_rows": len(train),
                "status": "insufficient_data",
                "train_end": train_end,
                "purged_through": cutoff_date,
                "target_horizon_days": target_horizon_days,
            },
        )

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]

    model = _get_model()
    model.fit(X_train, y_train)

    predict_rows = long_df[
        long_df["date"] == predict_date
    ].dropna(
        subset=FEATURE_COLS
    )

    if predict_rows.empty:
        tickers = long_df["ticker"].unique()

        return (
            pd.Series(0.0, index=tickers),
            {
                "n_train_rows": len(train),
                "status": "no_predict_rows",
                "train_end": train_end,
                "purged_through": cutoff_date,
                "target_horizon_days": target_horizon_days,
            },
        )

    preds = model.predict(predict_rows[FEATURE_COLS])

    pred_series = pd.Series(
        preds,
        index=predict_rows["ticker"].values,
        name="predicted_fwd_return_21d",
    )

    # In-sample IC is only a training diagnostic.
    train_preds = model.predict(X_train)
    train_ic, _ = spearmanr(train_preds, y_train)

    diagnostics = {
        "n_train_rows": len(train),
        "n_predicted": len(pred_series),
        "train_ic": train_ic,
        "status": "ok",
        "train_end": train_end,
        "purged_through": cutoff_date,
        "target_horizon_days": target_horizon_days,
    }

    return pred_series, diagnostics


# ------------------------------------------------------------------------------------
def historical_mean_baseline(
    long_df: pd.DataFrame,
    train_end: pd.Timestamp,
) -> pd.Series:
    """
    Historical-mean expected-return baseline.

    This uses only realized DAILY returns from dates strictly before train_end,
    so it does not require the 21-day target purge used by the ML model.
    """
    train = long_df[
        long_df["date"] < pd.Timestamp(train_end)
    ]

    mean_ret = (
        train.groupby("ticker")["returns"].mean()
        * 252
    )

    return mean_ret


# ------------------------------------------------------------------------------------
def out_of_sample_ic(
    predictions: pd.Series,
    actuals: pd.Series,
) -> float:
    """
    Spearman rank correlation between predicted and realized forward returns.
    """
    joined = pd.concat(
        [
            predictions.rename("pred"),
            actuals.rename("actual"),
        ],
        axis=1,
    ).dropna()

    if len(joined) < 5:
        return np.nan

    ic, _ = spearmanr(
        joined["pred"],
        joined["actual"],
    )

    return float(ic)


# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    print(
        "ML target:",
        TARGET_COL,
        "| horizon:",
        TARGET_HORIZON_DAYS,
        "trading days",
    )

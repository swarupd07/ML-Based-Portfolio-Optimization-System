import pandas as pd


def get_market_weights_asof(
    market_caps: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    tickers: pd.Index,
) -> pd.Series:

    available = market_caps[
        market_caps["date"] <= rebalance_date
    ].copy()

    available = available[
        available["ticker"].isin(tickers)
    ]

    latest = (
        available
        .sort_values("date")
        .groupby("ticker")
        .tail(1)
        .set_index("ticker")["market_cap"]
        .reindex(tickers)
    )

    latest = latest.dropna()

    if latest.empty or latest.sum() <= 0:
        raise ValueError(
            f"No valid market caps available at {rebalance_date}"
        )

    weights = latest / latest.sum()

    return weights


def align_shares_to_prices( shares_df: pd.DataFrame, prices: pd.DataFrame ) -> pd.DataFrame:

    shares_df = shares_df.copy()
    shares_df["date"] = pd.to_datetime(shares_df["date"]).astype("datetime64[ns]")

    records = []

    for ticker in prices.columns:

        ticker_shares = (
            shares_df[shares_df["ticker"] == ticker]
            [["date", "shares_outstanding"]]
            .sort_values("date"))

        price_dates = pd.DataFrame({"date": pd.to_datetime(prices.index).astype("datetime64[ns]")}).sort_values("date")

        aligned = pd.merge_asof(
            price_dates,
            ticker_shares,
            on="date",
            direction="backward",
        )

        aligned["ticker"] = ticker

        records.append(aligned)

    return pd.concat(records, ignore_index=True)

def build_market_caps(aligned_shares: pd.DataFrame, prices: pd.DataFrame ) -> pd.DataFrame:

    price_long = ( prices.stack().rename("price").reset_index())

    price_long.columns = ["date", "ticker", "price"]

    price_long["date"] = pd.to_datetime(
        price_long["date"]
    ).astype("datetime64[ns]")

    caps = aligned_shares.merge(
        price_long,
        on=["date", "ticker"],
        how="left",
    )

    caps["market_cap"] = (
        caps["price"]
        * caps["shares_outstanding"]
    )

    return caps


def estimate_delta_asof(
    market_returns: pd.Series,
    rebalance_date: pd.Timestamp,
    risk_free_rate: float = 0.065,
    lookback_days: int = 252,
) -> float:

    """
    Actual Estimated Deltas:

    2020-01-31 2.303
    2020-07-31 -0.612
    2021-01-31 0.937
    2021-07-31 10.857
    2022-01-31 4.239
    2022-07-31 1.195
    2023-01-31 -0.757
    2023-07-31 8.245
    2024-01-31 10.414
    2024-07-31 10.576
    2025-01-31 1.799
    2025-07-31 -3.049
    2026-01-31 2.186
    2026-07-31 -5.058
    
    """
    # estimated deltas are too noisy so fixing it to 2.5 

    window = (
        market_returns[
            market_returns.index < rebalance_date
        ]
        .dropna()
        .tail(lookback_days)
    )

    if len(window) < 60:
        raise ValueError(
            f"Not enough market history before {rebalance_date}"
        )

    ann_return = window.mean() * 252
    ann_variance = window.var() * 252

    if ann_variance <= 0:
        raise ValueError("Market variance must be positive.")

    delta = (
        ann_return - risk_free_rate
    ) / ann_variance

    return float(delta)
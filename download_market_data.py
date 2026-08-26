import pandas as pd
import yfinance as yf

"""tickers = [
    x.strip()
    for x in open("config/nifty50.txt")
    if x.strip()
]

records = []

for ticker in tickers:
    print("Downloading:", ticker)

    stock = yf.Ticker(ticker)

    shares = stock.get_shares_full(
        start="2019-01-01",
        end="2026-08-21",
    )

    if shares is None or len(shares) == 0:
        print("No shares data:", ticker)
        continue

    shares = shares.dropna()

    for date, value in shares.items():
        records.append(
            {
                "date": pd.Timestamp(date).tz_localize(None),
                "ticker": ticker,
                "shares_outstanding": float(value),
            }
        )

shares_df = pd.DataFrame(records)

shares_df.to_csv(
    "data/shares_outstanding.csv",
    index=False,
)

'''print(shares_df.head())
print(shares_df["ticker"].nunique())'''



market = yf.download(
    "^NSEI",
    start="2019-01-01",
    end="2026-08-21",
    auto_adjust=True,
    progress=False,
)

market = market[["Close"]].copy()
market.columns = ["close"]

market["return"] = market["close"].pct_change()

market.to_csv(
    "data/nifty50_market.csv"
)

print(market.head())"""

market = pd.read_csv(
    "data/nifty50_market.csv",
    index_col=0,
    parse_dates=True,
)

print(market.head())
print(market.tail())
print(market["return"].isna().mean())
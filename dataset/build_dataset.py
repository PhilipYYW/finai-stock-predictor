# dataset/build_dataset.py
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
import os

# ── Configuration ────────────────────────────────────────
SENTIMENT_PATH = "data/news_with_sentiment.csv"
OUTPUT_PATH    = "data/dataset.csv"

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
           "TSLA", "META", "JPM", "BAC", "AMD"]

# How many extra days to fetch for price lookahead
PRICE_LOOKBACK_DAYS = 365
# ─────────────────────────────────────────────────────────


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch daily closing prices for all tickers via yfinance"""
    print(f"Fetching price data from {start} to {end}...")
    all_prices = []

    for ticker in tqdm(tickers, desc="Downloading prices"):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"  [WARNING] No price data for {ticker}")
                continue

            df = df[["Close"]].copy()
            df.columns = ["close"]
            df["ticker"] = ticker
            df["date"]   = df.index.strftime("%Y-%m-%d")
            df = df.reset_index(drop=True)
            all_prices.append(df)

        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")

    if not all_prices:
        raise RuntimeError("No price data fetched. Check your internet connection.")

    prices = pd.concat(all_prices, ignore_index=True)
    print(f"Price data fetched: {len(prices)} rows across {len(tickers)} tickers\n")
    return prices


def compute_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute T+1, T+3, T+5 price change % and binary labels per ticker.
    Label: 1 = price UP, 0 = price DOWN or flat
    """
    result = []

    for ticker in prices["ticker"].unique():
        df = prices[prices["ticker"] == ticker].copy()
        df = df.sort_values("date").reset_index(drop=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        for horizon, col_pct, col_label in [
            (1, "price_change_1d", "label_1d"),
            (3, "price_change_3d", "label_3d"),
            (5, "price_change_5d", "label_5d"),
        ]:
            future_close  = df["close"].shift(-horizon)
            pct_change    = (future_close - df["close"]) / df["close"] * 100
            df[col_pct]   = pct_change.round(4)
            df[col_label] = (pct_change > 0).astype(int)

        result.append(df)

    return pd.concat(result, ignore_index=True)


def merge_sentiment_prices(sentiment: pd.DataFrame,
                           prices: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sentiment scores with price labels on (ticker, date).
    Each news article gets the corresponding price labels for that day.
    """
    # Keep only needed price columns
    price_cols = ["ticker", "date", "close",
                  "price_change_1d", "label_1d",
                  "price_change_3d", "label_3d",
                  "price_change_5d", "label_5d"]
    prices_slim = prices[price_cols].copy()

    merged = pd.merge(sentiment, prices_slim,
                      on=["ticker", "date"], how="inner")

    # Drop rows where any label is missing (end of price series)
    label_cols = ["label_1d", "label_3d", "label_5d"]
    before = len(merged)
    merged = merged.dropna(subset=label_cols).reset_index(drop=True)
    after  = len(merged)
    print(f"Dropped {before - after} rows with missing labels")

    # Convert label columns to int
    for col in label_cols:
        merged[col] = merged[col].astype(int)

    return merged


def save(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved to {path}")


if __name__ == "__main__":
    print("=" * 50)
    print("  Dataset Builder — T+1 / T+3 / T+5 Labels")
    print("=" * 50 + "\n")

    # 1. Load sentiment data
    if not os.path.exists(SENTIMENT_PATH):
        print(f"ERROR: {SENTIMENT_PATH} not found.")
        print("Please run nlp/finbert_scorer.py first.")
        exit(1)

    sentiment = pd.read_csv(SENTIMENT_PATH)
    print(f"Loaded {len(sentiment)} sentiment-scored articles")

    # Determine date range from sentiment data
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    min_date = sentiment["date"].min()
    max_date = sentiment["date"].max()

    # Fetch prices with extra buffer for T+5 lookahead
    start_date = (min_date - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date   = (max_date + timedelta(days=10)).strftime("%Y-%m-%d")

    sentiment["date"] = sentiment["date"].dt.strftime("%Y-%m-%d")

    # 2. Fetch price data
    prices = fetch_prices(TICKERS, start=start_date, end=end_date)

    # 3. Compute T+1 / T+3 / T+5 labels
    print("Computing price change labels...")
    prices_labeled = compute_labels(prices)
    print(f"Labels computed for {len(prices_labeled)} price rows\n")

    # 4. Merge sentiment + price labels
    print("Merging sentiment scores with price labels...")
    dataset = merge_sentiment_prices(sentiment, prices_labeled)
    print(f"Final dataset: {len(dataset)} rows\n")

    if dataset.empty:
        print("WARNING: Dataset is empty after merge.")
        print("This may happen if news dates don't match trading days.")
        exit(1)

    # 5. Save
    save(dataset, OUTPUT_PATH)

    # 6. Summary
    print("\n" + "=" * 50)
    print("  Dataset Summary")
    print("=" * 50)
    print(f"Total rows      : {len(dataset)}")
    print(f"Tickers         : {dataset['ticker'].nunique()}")
    print(f"Date range      : {dataset['date'].min()} to {dataset['date'].max()}")
    print(f"\nLabel distribution (T+1):")
    print(dataset["label_1d"].value_counts().rename({1: "UP", 0: "DOWN"}))
    print(f"\nLabel distribution (T+3):")
    print(dataset["label_3d"].value_counts().rename({1: "UP", 0: "DOWN"}))
    print(f"\nLabel distribution (T+5):")
    print(dataset["label_5d"].value_counts().rename({1: "UP", 0: "DOWN"}))
    print(f"\nAverage sentiment score : {dataset['sentiment_score'].mean():.4f}")
    print(f"\nPreview (first 5 rows):")
    preview_cols = ["ticker", "date", "headline",
                    "sentiment_label", "sentiment_score",
                    "price_change_1d", "label_1d",
                    "price_change_3d", "label_3d",
                    "price_change_5d", "label_5d"]
    print(dataset[preview_cols].head().to_string())
    print("\nDone!")

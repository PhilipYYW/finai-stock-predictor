# dataset/feature_engineering.py
import pandas as pd
import numpy as np
import os

# ── Configuration ────────────────────────────────────────
INPUT_PATH  = "data/dataset.csv"
OUTPUT_PATH = "data/features.csv"
# ─────────────────────────────────────────────────────────


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sentiment-based rolling features per ticker"""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        s    = df.loc[mask, "sentiment_score"]

        # Rolling mean sentiment (3-day, 5-day)
        df.loc[mask, "sent_roll3"] = s.rolling(3, min_periods=1).mean()
        df.loc[mask, "sent_roll5"] = s.rolling(5, min_periods=1).mean()

        # Sentiment momentum: today vs 3-day average
        df.loc[mask, "sent_momentum"] = s - s.rolling(3, min_periods=1).mean()

        # Sentiment std (volatility of sentiment)
        df.loc[mask, "sent_std5"] = s.rolling(5, min_periods=1).std().fillna(0)

        # Positive / negative ratio over 5 days
        pos = (s > 0).rolling(5, min_periods=1).sum()
        neg = (s < 0).rolling(5, min_periods=1).sum()
        df.loc[mask, "pos_neg_ratio"] = (pos - neg) / 5

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price-based technical features per ticker"""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    for ticker in df["ticker"].unique():
        mask  = df["ticker"] == ticker
        close = df.loc[mask, "close"]

        # Price momentum (1-day, 5-day return)
        df.loc[mask, "price_mom1"] = close.pct_change(1).fillna(0)
        df.loc[mask, "price_mom5"] = close.pct_change(5).fillna(0)

        # Volatility (5-day std of returns)
        ret = close.pct_change()
        df.loc[mask, "volatility5"] = ret.rolling(5, min_periods=1).std().fillna(0)

        # Simple moving averages
        df.loc[mask, "sma5"]  = close.rolling(5,  min_periods=1).mean()
        df.loc[mask, "sma10"] = close.rolling(10, min_periods=1).mean()

        # SMA crossover signal: 1 if price > sma10
        df.loc[mask, "above_sma10"] = (close > df.loc[mask, "sma10"]).astype(int)

        # Distance from SMA10 (%)
        df.loc[mask, "dist_sma10"] = (
            (close - df.loc[mask, "sma10"]) / df.loc[mask, "sma10"]
        ).fillna(0)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features"""
    dt = pd.to_datetime(df["date"])
    df["day_of_week"] = dt.dt.dayofweek        # 0=Mon, 4=Fri
    df["month"]       = dt.dt.month
    df["is_monday"]   = (dt.dt.dayofweek == 0).astype(int)
    df["is_friday"]   = (dt.dt.dayofweek == 4).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ticker and sentiment label as integers"""
    df["ticker_enc"] = pd.Categorical(df["ticker"]).codes
    label_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_enc"] = df["sentiment_label"].map(label_map).fillna(0).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_sentiment_features(df)
    df = add_price_features(df)
    df = add_calendar_features(df)
    df = encode_categoricals(df)
    return df


FEATURE_COLS = [
    # Sentiment features
    "sentiment_score", "sentiment_enc",
    "sent_roll3", "sent_roll5", "sent_momentum", "sent_std5",
    "pos_neg_ratio",
    # Price features
    "price_mom1", "price_mom5", "volatility5",
    "above_sma10", "dist_sma10",
    # Calendar features
    "day_of_week", "is_monday", "is_friday",
    # Categorical
    "ticker_enc",
]

LABEL_COLS = ["label_1d", "label_3d", "label_5d"]


if __name__ == "__main__":
    print("=" * 50)
    print("  Feature Engineering")
    print("=" * 50 + "\n")

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found.")
        print("Please run dataset/build_dataset.py first.")
        exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows\n")

    df = build_features(df)

    # Drop rows with any NaN in feature or label columns
    all_cols = FEATURE_COLS + LABEL_COLS
    before   = len(df)
    df       = df.dropna(subset=all_cols).reset_index(drop=True)
    after    = len(df)
    print(f"Dropped {before - after} rows with NaN -> {after} rows remaining")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved to {OUTPUT_PATH}")

    print(f"\nFeature columns ({len(FEATURE_COLS)}):")
    for col in FEATURE_COLS:
        print(f"  {col:25s}  mean={df[col].mean():.4f}  std={df[col].std():.4f}")

    print(f"\nLabel distribution (T+1): UP={df['label_1d'].sum()}  DOWN={(df['label_1d']==0).sum()}")
    print(f"Label distribution (T+3): UP={df['label_3d'].sum()}  DOWN={(df['label_3d']==0).sum()}")
    print(f"Label distribution (T+5): UP={df['label_5d'].sum()}  DOWN={(df['label_5d']==0).sum()}")
    print("\nDone!")

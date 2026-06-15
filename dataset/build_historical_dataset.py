# dataset/build_historical_dataset.py
"""
Builds a 2-year dataset using:
1. yfinance OHLCV price history (2 years)
2. Technical indicators as primary features
3. Sentiment proxy model trained on real FinBERT scores,
   then applied to historical dates
4. T+1 / T+3 / T+5 labels from actual price data
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── Configuration ────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
           "TSLA", "META", "JPM", "BAC", "AMD"]

REAL_SENTIMENT_PATH = "data/news_with_sentiment.csv"
OUTPUT_PATH         = "data/dataset_historical.csv"

YEARS_BACK = 2
# ─────────────────────────────────────────────────────────


# ── Step 1: Fetch 2-year OHLCV ───────────────────────────
def fetch_ohlcv(tickers: list[str]) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=365 * YEARS_BACK + 10)

    print(f"Fetching OHLCV: {start.strftime('%Y-%m-%d')} "
          f"to {end.strftime('%Y-%m-%d')}")

    frames = []
    for ticker in tickers:
        df = yf.download(ticker,
                         start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         auto_adjust=True,
                         progress=False)
        if df.empty:
            print(f"  [WARNING] No data for {ticker}")
            continue

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df["ticker"] = ticker
        df["date"]   = df.index.strftime("%Y-%m-%d")
        df = df.reset_index(drop=True)
        frames.append(df)
        print(f"  {ticker}: {len(df)} trading days")

    return pd.concat(frames, ignore_index=True)


# ── Step 2: Technical indicators ─────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = []

    for ticker in df["ticker"].unique():
        t = df[df["ticker"] == ticker].copy().sort_values("date").reset_index(drop=True)
        c = t["close"]
        v = t["volume"]

        # Returns
        t["ret_1d"] = c.pct_change(1)
        t["ret_3d"] = c.pct_change(3)
        t["ret_5d"] = c.pct_change(5)

        # Moving averages
        t["sma5"]   = c.rolling(5).mean()
        t["sma10"]  = c.rolling(10).mean()
        t["sma20"]  = c.rolling(20).mean()

        # EMA
        t["ema12"]  = c.ewm(span=12, adjust=False).mean()
        t["ema26"]  = c.ewm(span=26, adjust=False).mean()

        # MACD
        t["macd"]        = t["ema12"] - t["ema26"]
        t["macd_signal"] = t["macd"].ewm(span=9, adjust=False).mean()
        t["macd_hist"]   = t["macd"] - t["macd_signal"]

        # RSI (14)
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        t["rsi14"] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20, 2)
        bb_mid        = c.rolling(20).mean()
        bb_std        = c.rolling(20).std()
        t["bb_upper"] = bb_mid + 2 * bb_std
        t["bb_lower"] = bb_mid - 2 * bb_std
        t["bb_pct"]   = (c - t["bb_lower"]) / (t["bb_upper"] - t["bb_lower"] + 1e-9)

        # Volatility
        t["vol5"]  = t["ret_1d"].rolling(5).std()
        t["vol20"] = t["ret_1d"].rolling(20).std()

        # Volume features
        t["vol_ratio"] = v / v.rolling(20).mean()

        # Distance from SMAs
        t["dist_sma5"]  = (c - t["sma5"])  / t["sma5"]
        t["dist_sma20"] = (c - t["sma20"]) / t["sma20"]

        # SMA crossover signals
        t["above_sma5"]  = (c > t["sma5"]).astype(int)
        t["above_sma20"] = (c > t["sma20"]).astype(int)
        t["sma5_above_sma20"] = (t["sma5"] > t["sma20"]).astype(int)

        # Calendar
        dt = pd.to_datetime(t["date"])
        t["day_of_week"] = dt.dt.dayofweek
        t["month"]       = dt.dt.month
        t["is_monday"]   = (dt.dt.dayofweek == 0).astype(int)
        t["is_friday"]   = (dt.dt.dayofweek == 4).astype(int)

        result.append(t)

    return pd.concat(result, ignore_index=True)


# ── Step 3: T+1 / T+3 / T+5 labels ──────────────────────
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for ticker in df["ticker"].unique():
        t = df[df["ticker"] == ticker].copy().sort_values("date").reset_index(drop=True)
        c = t["close"]
        for h, pc, lc in [(1, "price_change_1d", "label_1d"),
                          (3, "price_change_3d", "label_3d"),
                          (5, "price_change_5d", "label_5d")]:
            fut       = c.shift(-h)
            pct       = (fut - c) / c * 100
            t[pc]     = pct.round(4)
            t[lc]     = (pct > 0).astype(int)
        result.append(t)
    return pd.concat(result, ignore_index=True)


# ── Step 4: Sentiment proxy model ────────────────────────
def build_sentiment_proxy(df: pd.DataFrame,
                          real_sentiment_path: str) -> pd.DataFrame:
    """
    Train a simple Ridge regression on real FinBERT scores
    using technical features as predictors.
    Apply to all historical dates.
    """
    if not os.path.exists(real_sentiment_path):
        print("  No real sentiment data found — using neutral (0.0)")
        df["sentiment_score"] = 0.0
        df["sentiment_label"] = "neutral"
        return df

    real = pd.read_csv(real_sentiment_path)
    real["date"] = pd.to_datetime(real["date"]).dt.strftime("%Y-%m-%d")

    # Average sentiment per (ticker, date)
    sent_agg = (real.groupby(["ticker", "date"])["sentiment_score"]
                .mean().reset_index())

    # Merge with price data to get technical features on those dates
    proxy_features = ["ret_1d", "ret_3d", "ret_5d", "rsi14",
                      "macd", "bb_pct", "vol5", "dist_sma20",
                      "vol_ratio", "day_of_week", "month"]

    merged = pd.merge(sent_agg, df[["ticker", "date"] + proxy_features],
                      on=["ticker", "date"], how="inner").dropna()

    if len(merged) < 10:
        print(f"  Only {len(merged)} matched rows — using neutral sentiment")
        df["sentiment_score"] = 0.0
        df["sentiment_label"] = "neutral"
        return df

    print(f"  Training sentiment proxy on {len(merged)} real data points")

    X_real = merged[proxy_features].values
    y_real = merged["sentiment_score"].values

    scaler = StandardScaler()
    X_real = scaler.fit_transform(X_real)

    proxy  = Ridge(alpha=1.0)
    proxy.fit(X_real, y_real)

    # Apply to all historical data
    feat_data = df[proxy_features].copy().fillna(0)
    X_all     = scaler.transform(feat_data.values)
    df["sentiment_score"] = proxy.predict(X_all).round(4)

    # Clip to [-1, 1]
    df["sentiment_score"] = df["sentiment_score"].clip(-1, 1)

    # Derive label
    df["sentiment_label"] = pd.cut(
        df["sentiment_score"],
        bins=[-1.01, -0.1, 0.1, 1.01],
        labels=["negative", "neutral", "positive"]
    ).astype(str)

    # Save proxy model
    os.makedirs("models", exist_ok=True)
    joblib.dump(proxy,  "models/sentiment_proxy.pkl")
    joblib.dump(scaler, "models/sentiment_proxy_scaler.pkl")
    print(f"  Sentiment proxy saved to models/")

    return df


# ── Main ─────────────────────────────────────────────────
FEATURE_COLS_HIST = [
    "ret_1d", "ret_3d", "ret_5d",
    "rsi14", "macd", "macd_signal", "macd_hist",
    "bb_pct", "vol5", "vol20",
    "dist_sma5", "dist_sma20",
    "above_sma5", "above_sma20", "sma5_above_sma20",
    "vol_ratio",
    "sentiment_score",
    "day_of_week", "month", "is_monday", "is_friday",
]

LABEL_COLS = ["label_1d", "label_3d", "label_5d"]


if __name__ == "__main__":
    print("=" * 55)
    print("  Historical Dataset Builder (2 Years)")
    print("=" * 55 + "\n")

    # 1. OHLCV
    df = fetch_ohlcv(TICKERS)
    print(f"\nTotal OHLCV rows: {len(df)}\n")

    # 2. Technical indicators
    print("Computing technical indicators...")
    df = add_technical_indicators(df)

    # 3. Labels
    print("Computing T+1 / T+3 / T+5 labels...")
    df = add_labels(df)

    # 4. Sentiment proxy
    print("\nBuilding sentiment proxy model...")
    df = build_sentiment_proxy(df, REAL_SENTIMENT_PATH)

    # 5. Encode ticker
    df["ticker_enc"] = pd.Categorical(df["ticker"]).codes

    # 6. Drop NaN rows
    all_cols = FEATURE_COLS_HIST + LABEL_COLS + ["ticker_enc"]
    before   = len(df)
    df       = df.dropna(subset=all_cols).reset_index(drop=True)
    after    = len(df)
    print(f"\nDropped {before - after} rows with NaN -> {after} rows remaining")

    # 7. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved to {OUTPUT_PATH}")

    # 8. Summary
    print(f"\n{'='*55}")
    print("  Dataset Summary")
    print(f"{'='*55}")
    print(f"Total rows   : {len(df)}")
    print(f"Tickers      : {df['ticker'].nunique()}")
    print(f"Date range   : {df['date'].min()} to {df['date'].max()}")
    print(f"Features     : {len(FEATURE_COLS_HIST) + 1}")
    print(f"\nLabel distribution (T+1):")
    vc = df["label_1d"].value_counts().rename({1: "UP", 0: "DOWN"})
    print(vc.to_string())
    print(f"\nLabel distribution (T+3):")
    vc = df["label_3d"].value_counts().rename({1: "UP", 0: "DOWN"})
    print(vc.to_string())
    print(f"\nLabel distribution (T+5):")
    vc = df["label_5d"].value_counts().rename({1: "UP", 0: "DOWN"})
    print(vc.to_string())
    print("\nDone!")

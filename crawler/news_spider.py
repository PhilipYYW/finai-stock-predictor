# crawler/news_spider.py
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import time
import os

# ── Configuration ────────────────────────────────────────
import json as _json

def _load_tickers():
    default = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
               "TSLA", "META", "JPM", "BAC", "AMD"]
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    root_dir    = os.path.dirname(script_dir)   # one level up from crawler/
    config_path = os.path.join(root_dir, "config.json")

    print(f"  Looking for config.json at: {config_path}")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = _json.load(f)
            tickers = cfg.get("tickers", default)
            print(f"  Loaded {len(tickers)} tickers: {tickers}")
            return tickers
        except Exception as e:
            print(f"  Warning: {e}, using defaults")
    else:
        print(f"  config.json not found, using defaults")
    return default

TICKERS = _load_tickers()

# Default curated queries for known tickers
_DEFAULT_QUERIES = {
    "AAPL":  "Apple AAPL stock earnings",
    "MSFT":  "Microsoft MSFT stock earnings",
    "GOOGL": "Google Alphabet GOOGL stock",
    "AMZN":  "Amazon AMZN stock earnings",
    "NVDA":  "Nvidia NVDA stock earnings",
    "TSLA":  "Tesla TSLA stock earnings",
    "META":  "Meta Platforms META stock",
    "JPM":   "JPMorgan JPM stock earnings",
    "BAC":   "Bank of America BAC stock",
    "AMD":   "AMD semiconductor stock",
}

def _build_ticker_queries(tickers: list) -> dict:
    """
    Build search queries for all tickers.
    Use curated query if available, otherwise auto-generate from
    company name in config.json or fall back to "{TICKER} stock".
    """
    # Load company names from config.json
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    root_dir    = os.path.dirname(script_dir)
    config_path = os.path.join(root_dir, "config.json")
    ticker_names = {}
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = _json.load(f)
            ticker_names = cfg.get("ticker_names", {})
        except Exception:
            pass

    queries = {}
    for t in tickers:
        if t in _DEFAULT_QUERIES:
            queries[t] = _DEFAULT_QUERIES[t]
        elif t in ticker_names and ticker_names[t]:
            queries[t] = f"{ticker_names[t]} {t} stock earnings"
        else:
            queries[t] = f"{t} stock earnings"
    return queries

TICKER_QUERIES = _build_ticker_queries(TICKERS)

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "f3829837585746d3a4a4b5d5b1a4130a")
OUTPUT_PATH  = "data/raw_news.csv"
DAYS_BACK    = 29
PAGE_SIZE    = 100
# ─────────────────────────────────────────────────────────


def fetch_newsapi(ticker: str, query: str,
                  from_date: str, to_date: str) -> list[dict]:
    """Fetch news for one ticker from NewsAPI.org"""
    # Use ticker itself as fallback query if not in TICKER_QUERIES
    if not query or query == ticker:
        query = f"{ticker} stock"

    url    = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "from":     from_date,
        "to":       to_date,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": PAGE_SIZE,
        "apiKey":   NEWS_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if data.get("status") != "ok":
            print(f"  [API ERROR] {ticker}: {data.get('message')}")
            return []

        articles = []
        for item in data.get("articles", []):
            pub_raw  = item.get("publishedAt", "")
            try:
                dt       = datetime.strptime(pub_raw, "%Y-%m-%dT%H:%M:%SZ")
                pub_str  = dt.strftime("%Y-%m-%d %H:%M:%S")
                pub_date = dt.strftime("%Y-%m-%d")
            except Exception:
                pub_str = pub_date = ""

            headline = (item.get("title") or "").strip()
            summary  = (item.get("description") or "").strip()

            if not headline or headline == "[Removed]" or not pub_date:
                continue

            # Tag each article with its correct ticker
            articles.append({
                "ticker":   ticker,
                "date":     pub_date,
                "datetime": pub_str,
                "headline": headline,
                "summary":  summary,
                "link":     item.get("url", ""),
                "source":   item.get("source", {}).get("name", ""),
            })

        return articles

    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return []


def crawl_all(tickers: list[str]) -> pd.DataFrame:
    """Crawl past 29 days of news for all tickers separately"""
    today     = datetime.now(tz=timezone.utc)
    from_date = (today - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    to_date   = today.strftime("%Y-%m-%d")

    print(f"Date range: {from_date} to {to_date}\n")

    all_articles = []
    for ticker in tqdm(tickers, desc="Fetching news"):
        query    = TICKER_QUERIES.get(ticker, ticker)
        articles = fetch_newsapi(ticker, query, from_date, to_date)
        print(f"  {ticker}: {len(articles)} articles")
        all_articles.extend(articles)
        time.sleep(1.0)   # Respect rate limit (100 req/day free plan)

    df = pd.DataFrame(all_articles)

    if df.empty:
        print("WARNING: No articles fetched.")
        return df

    # De-duplicate within each ticker separately
    deduped = []
    for ticker in df["ticker"].unique():
        sub = df[df["ticker"] == ticker].drop_duplicates(subset=["headline"])
        deduped.append(sub)
    df = pd.concat(deduped, ignore_index=True)

    df = df[df["headline"].str.len() > 0].reset_index(drop=True)
    df = df.sort_values(["ticker", "datetime"],
                        ascending=[True, False]).reset_index(drop=True)

    return df


def save(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved to {path}")


if __name__ == "__main__":
    print("=" * 50)
    print("  NewsAPI Historical News Crawler")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print("=" * 50 + "\n")

    df = crawl_all(TICKERS)

    if not df.empty:
        save(df, OUTPUT_PATH)
        print(f"\nDate range  : {df['date'].min()} to {df['date'].max()}")
        print(f"Total       : {len(df)} articles")
        print(f"\nPer ticker  :")
        print(df["ticker"].value_counts().to_string())
        print(f"\nPreview (first 5 rows):")
        print(df[["ticker", "date", "headline"]].head().to_string())

# crawler/news_spider.py
import feedparser
import pandas as pd
import requests
from datetime import datetime, timezone
from tqdm import tqdm
import time
import os

# ── Configuration ────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
           "TSLA", "META", "JPM", "BAC", "AMD"]

OUTPUT_PATH = "data/raw_news.csv"
# ─────────────────────────────────────────────────────────

RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

def fetch_rss(ticker: str) -> list[dict]:
    """Fetch Yahoo Finance RSS news for a single ticker"""
    url = RSS_TEMPLATE.format(ticker=ticker)
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries:
            # Parse publication time
            published = entry.get("published", "")
            try:
                pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                pub_str = pub_dt.strftime("%Y-%m-%d %H:%M:%S")
                pub_date = pub_dt.strftime("%Y-%m-%d")
            except Exception:
                pub_str = ""
                pub_date = ""

            articles.append({
                "ticker":    ticker,
                "date":      pub_date,
                "datetime":  pub_str,
                "headline":  entry.get("title", "").strip(),
                "summary":   entry.get("summary", "").strip(),
                "link":      entry.get("link", "").strip(),
                "source":    "Yahoo Finance RSS",
            })
        return articles

    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return []


def crawl_all(tickers: list[str]) -> pd.DataFrame:
    """Crawl news for all tickers and return a DataFrame"""
    all_articles = []

    for ticker in tqdm(tickers, desc="Fetching news"):
        articles = fetch_rss(ticker)
        all_articles.extend(articles)
        time.sleep(0.5)          # Avoid rate limiting

    df = pd.DataFrame(all_articles)

    if df.empty:
        print("WARNING: No articles fetched. Please check your internet connection.")
        return df

    # Remove duplicate headlines
    before = len(df)
    df = df.drop_duplicates(subset=["headline"])
    after = len(df)
    print(f"\nDe-duplicated: {before} -> {after} articles")

    # Remove empty headlines
    df = df[df["headline"].str.len() > 0].reset_index(drop=True)

    return df


def save(df: pd.DataFrame, path: str):
    """Save to CSV, merging with existing file if present"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["headline"]).reset_index(drop=True)
        print(f"Merged with existing data: {len(df)} total articles")

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved to {path}")


if __name__ == "__main__":
    print("=" * 50)
    print("  Yahoo Finance News Crawler")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print("=" * 50)

    df = crawl_all(TICKERS)

    if not df.empty:
        save(df, OUTPUT_PATH)
        print(f"\nPreview (first 5 rows):")
        print(df[["ticker", "date", "headline"]].head())
        print(f"\nTotal: {len(df)} articles collected")

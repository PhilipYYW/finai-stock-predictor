# crawler/news_spider.py
import feedparser
import pandas as pd
import requests
from datetime import datetime, timezone
from tqdm import tqdm
import time
import os

# ── 設定區 ──────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
           "TSLA", "META", "JPM", "BAC", "AMD"]

OUTPUT_PATH = "data/raw_news.csv"
# ────────────────────────────────────────────────────────

RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

def fetch_rss(ticker: str) -> list[dict]:
    """抓取單一股票的 Yahoo Finance RSS 新聞"""
    url = RSS_TEMPLATE.format(ticker=ticker)
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries:
            # 解析發布時間
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
    """爬取所有股票新聞，回傳 DataFrame"""
    all_articles = []

    for ticker in tqdm(tickers, desc="爬取新聞"):
        articles = fetch_rss(ticker)
        all_articles.extend(articles)
        time.sleep(0.5)          # 避免太快被封鎖

    df = pd.DataFrame(all_articles)

    if df.empty:
        print("⚠️  沒有抓到任何新聞，請檢查網路連線")
        return df

    # 去除重複標題
    before = len(df)
    df = df.drop_duplicates(subset=["headline"])
    after = len(df)
    print(f"\n✅ 去重：{before} → {after} 筆")

    # 移除空標題
    df = df[df["headline"].str.len() > 0].reset_index(drop=True)

    return df


def save(df: pd.DataFrame, path: str):
    """儲存 CSV，若檔案已存在則合併去重"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["headline"]).reset_index(drop=True)
        print(f"📂 合併既有資料，共 {len(df)} 筆")

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"💾 已儲存至 {path}")


if __name__ == "__main__":
    print("=" * 50)
    print("  Yahoo Finance 新聞爬蟲")
    print(f"  目標股票：{', '.join(TICKERS)}")
    print("=" * 50)

    df = crawl_all(TICKERS)

    if not df.empty:
        save(df, OUTPUT_PATH)
        print(f"\n📊 資料預覽（前 5 筆）：")
        print(df[["ticker", "date", "headline"]].head())
        print(f"\n總計：{len(df)} 筆新聞")

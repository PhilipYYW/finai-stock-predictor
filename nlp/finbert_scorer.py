# nlp/finbert_scorer.py
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# ── 設定區 ──────────────────────────────────────────────
INPUT_PATH  = "data/raw_news.csv"
OUTPUT_PATH = "data/news_with_sentiment.csv"
MODEL_NAME  = "ProsusAI/finbert"
BATCH_SIZE  = 16          # RTX 3060 6GB 跑 16 很穩，可調高到 32
MAX_LENGTH  = 512
# ────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    "positive":  1.0,
    "negative": -1.0,
    "neutral":   0.0,
}


def load_model():
    """載入 FinBERT tokenizer 與模型"""
    print(f"🔧 載入 FinBERT（{MODEL_NAME}）...")
    print(f"⚡ 使用裝置：{DEVICE.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    print("✅ 模型載入完成\n")
    return tokenizer, model


def predict_batch(texts: list[str], tokenizer, model) -> list[dict]:
    """對一個 batch 的文字做情緒預測"""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu()

    # FinBERT 輸出順序：positive / negative / neutral
    labels = ["positive", "negative", "neutral"]
    results = []
    for prob in probs:
        idx = prob.argmax().item()
        label = labels[idx]
        results.append({
            "sentiment_label": label,
            "sentiment_score": round(LABEL_MAP[label] * prob[idx].item(), 4),
            "prob_positive":   round(prob[0].item(), 4),
            "prob_negative":   round(prob[1].item(), 4),
            "prob_neutral":    round(prob[2].item(), 4),
        })
    return results


def score_dataframe(df: pd.DataFrame, tokenizer, model) -> pd.DataFrame:
    """對整個 DataFrame 做情緒分析"""
    # 合併 headline + summary 當輸入文字（給模型更多上下文）
    texts = (df["headline"] + ". " + df["summary"].fillna("")).tolist()

    all_results = []
    batches = [texts[i:i+BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    for batch in tqdm(batches, desc="FinBERT 分析中"):
        results = predict_batch(batch, tokenizer, model)
        all_results.extend(results)

    sentiment_df = pd.DataFrame(all_results)
    return pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)


def save(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"💾 已儲存至 {path}")


if __name__ == "__main__":
    print("=" * 50)
    print("  FinBERT 情緒分析")
    print("=" * 50)

    # 讀取爬蟲輸出
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 找不到 {INPUT_PATH}，請先執行 crawler/news_spider.py")
        exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"📂 讀取 {len(df)} 筆新聞\n")

    # 載入模型並分析
    tokenizer, model = load_model()
    df_scored = score_dataframe(df, tokenizer, model)

    # 儲存結果
    save(df_scored, OUTPUT_PATH)

    # 預覽結果
    print(f"\n📊 情緒分布：")
    print(df_scored["sentiment_label"].value_counts())
    print(f"\n📊 資料預覽（前 5 筆）：")
    print(df_scored[["ticker", "date", "headline", "sentiment_label", "sentiment_score"]].head())
    print(f"\n✅ 完成！共 {len(df_scored)} 筆新聞已標注情緒")

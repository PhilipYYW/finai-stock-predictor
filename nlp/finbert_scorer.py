# nlp/finbert_scorer.py
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# ── Configuration ────────────────────────────────────────
INPUT_PATH  = "data/raw_news.csv"
OUTPUT_PATH = "data/news_with_sentiment.csv"
MODEL_NAME  = "ProsusAI/finbert"
BATCH_SIZE  = 16          # Stable for RTX 3060 6GB, can increase to 32
MAX_LENGTH  = 512
# ─────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    "positive":  1.0,
    "negative": -1.0,
    "neutral":   0.0,
}


def load_model():
    """Load FinBERT tokenizer and model"""
    print(f"Loading FinBERT ({MODEL_NAME})...")
    print(f"Device: {DEVICE.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully\n")
    return tokenizer, model


def predict_batch(texts: list[str], tokenizer, model) -> list[dict]:
    """Run sentiment prediction on a batch of texts"""
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

    # FinBERT output order: positive / negative / neutral
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
    """Run sentiment analysis on the entire DataFrame"""
    # Combine headline + summary for richer context
    texts = (df["headline"] + ". " + df["summary"].fillna("")).tolist()

    all_results = []
    batches = [texts[i:i+BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    for batch in tqdm(batches, desc="Running FinBERT"):
        results = predict_batch(batch, tokenizer, model)
        all_results.extend(results)

    sentiment_df = pd.DataFrame(all_results)
    return pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)


def save(df: pd.DataFrame, path: str):
    """Save results to CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved to {path}")


if __name__ == "__main__":
    print("=" * 50)
    print("  FinBERT Sentiment Scorer")
    print("=" * 50)

    # Load crawler output
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found. Please run crawler/news_spider.py first.")
        exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} articles\n")

    # Load model and run analysis
    tokenizer, model = load_model()
    df_scored = score_dataframe(df, tokenizer, model)

    # Save results
    save(df_scored, OUTPUT_PATH)

    # Preview results
    print(f"\nSentiment distribution:")
    print(df_scored["sentiment_label"].value_counts())
    print(f"\nPreview (first 5 rows):")
    print(df_scored[["ticker", "date", "headline", "sentiment_label", "sentiment_score"]].head())
    print(f"\nDone! {len(df_scored)} articles scored.")

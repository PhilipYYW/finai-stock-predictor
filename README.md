# 📈 FinAI Stock Predictor

> An end-to-end AI pipeline that analyzes financial news sentiment and predicts stock price movements — powered by FinBERT, XGBoost, and LSTM.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00C896)

---

## 🧠 What It Does

```
Financial News (NewsAPI)
        ↓
FinBERT Sentiment Analysis    ← finance-domain NLP model
        ↓
2-Year Historical Dataset     ← price + sentiment + technical indicators
        ↓
XGBoost + LSTM Training       ← T+1 / T+3 / T+5 prediction
        ↓
4-Strategy Backtesting        ← vs Buy & Hold benchmark
        ↓
Streamlit Dashboard           ← real-time signals + heatmap + analysis
```

---

## 🖥️ Dashboard Preview

| Page | Content |
|------|---------|
| **Dashboard** | Live signals for all stocks · Sentiment heatmap · Pipeline controls |
| **Analysis** | Backtest results · XGBoost vs LSTM comparison |
| **Deep Dive** | Individual stock: price chart · RSI · sentiment · prediction gauge |

---

## ⚙️ Quick Start

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 – 3.12 | **Not 3.13+** (PyTorch not supported yet) |
| NVIDIA GPU | Optional | CPU works, GPU recommended for FinBERT speed |
| NewsAPI Key | Free | 100 requests/day — [get one here](https://newsapi.org) |

---

### 1. Clone the repo

```bash
git clone https://github.com/PhilipYYW/finai-stock-predictor.git
cd finai-stock-predictor
```

### 2. Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch

**With NVIDIA GPU (recommended):**
```bash
# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**CPU only:**
```bash
pip install torch torchvision torchaudio
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set your NewsAPI key

```bash
# Windows
echo NEWS_API_KEY=your_key_here > .env

# Mac / Linux
echo "NEWS_API_KEY=your_key_here" > .env
```

### 6. Run the full pipeline

```bash
python crawler/news_spider.py          # Fetch financial news
python nlp/finbert_scorer.py           # FinBERT sentiment scoring
python dataset/build_historical_dataset.py  # Build 2-year dataset
python models/train_xgboost_v2.py     # Train XGBoost models
python models/train_lstm.py           # Train LSTM models
python backtest/run_backtest.py       # Run 4-strategy backtest
```

### 7. Launch dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

> 💡 **After the first setup**, you can use the **Run Full Pipeline** button inside the dashboard — no terminal needed.

---

## 📊 Model Performance

Evaluated on out-of-sample test set (975 rows):

| Model | T+1 AUC | T+3 AUC | T+5 AUC |
|-------|---------|---------|---------|
| XGBoost v2 | 0.542 | **0.567** | 0.569 |
| LSTM | **0.560** | 0.554 | **0.578** |

> 📌 52–58% AUC is considered strong for financial prediction. All models beat the random 50% baseline.

---

## 📈 Backtest Results

| Strategy | Return | Sharpe | Max Drawdown | Win Rate |
|----------|--------|--------|-------------|---------|
| Sentiment Long/Short | +288.5% | 11.02 | -5.3% | 84.0% |
| Momentum + Sentiment | +89.7% | 6.50 | -3.5% | 86.0% |
| Mean Reversion | -4.5% | -0.10 | -6.1% | 26.7% |
| SNR + Sentiment | +23.6% | 4.29 | **-0.6%** | **88.7%** |
| Buy & Hold (Benchmark) | +11.0% | 1.31 | -15.0% | — |

> ⚠️ Returns use a sentiment proxy model and may overstate real performance. See disclaimer in dashboard.

---

## 🗂️ Project Structure

```
finai-stock-predictor/
├── crawler/
│   └── news_spider.py              # NewsAPI crawler (dynamic tickers)
├── nlp/
│   └── finbert_scorer.py           # FinBERT sentiment scoring
├── dataset/
│   ├── feature_engineering.py      # Technical + sentiment features
│   └── build_historical_dataset.py # 2-year OHLCV + sentiment dataset
├── models/
│   ├── train_xgboost_v2.py         # XGBoost trainer (T+1/T+3/T+5)
│   └── train_lstm.py               # LSTM trainer (sequential)
├── strategy/
│   └── indicators.py               # Pivot points, S/R levels
├── backtest/
│   └── run_backtest.py             # 4-strategy backtesting engine
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── config.json                     # Ticker configuration
├── requirements.txt
└── README.md
```

---

## ➕ Adding New Stocks

1. Open the dashboard → **Dashboard** page
2. Bottom section → **Add New Stock** → enter ticker + company name
3. Click **Run Full Pipeline**
4. New stock appears with real predictions in ~5 minutes

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| NLP | [FinBERT](https://huggingface.co/ProsusAI/finbert) (HuggingFace) |
| Deep Learning | PyTorch 2.6 (CUDA) |
| ML | XGBoost, scikit-learn |
| Data | yfinance, NewsAPI, Pandas |
| Technical Analysis | RSI, MACD, Bollinger Bands, SMA, Pivot Points |
| Dashboard | Streamlit, Plotly |
| Version Control | Git, GitHub |

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**.
It is not financial advice. Past backtest performance does not guarantee future results.
All data sourced from Yahoo Finance and NewsAPI for research use only.

---

## 📄 License

MIT License — free to use for personal and educational purposes.

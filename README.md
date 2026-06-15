# 📈 FinAI Stock Predictor

> An end-to-end AI pipeline that analyzes financial news sentiment and predicts stock price movements using FinBERT, XGBoost, and LSTM.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 How It Works

```
Financial News (NewsAPI)
        ↓
FinBERT Sentiment Analysis
        ↓
Dataset Construction (T+1 / T+3 / T+5 Labels)
        ↓
XGBoost + LSTM Training
        ↓
Backtesting (4 Strategies)
        ↓
Streamlit Dashboard
```

The system collects financial news, scores each article with [FinBERT](https://huggingface.co/ProsusAI/finbert) (a finance-domain BERT model), merges sentiment scores with 2-year historical price data, and trains models to predict whether a stock will go **UP** or **DOWN** over the next 1, 3, and 5 trading days.

---

## 🚀 Features

| Feature | Details |
|---------|---------|
| **News Crawler** | NewsAPI integration, dynamic ticker management |
| **Sentiment Analysis** | FinBERT (ProsusAI) — finance-specific NLP |
| **Prediction Horizons** | T+1, T+3, T+5 (tomorrow, 3-day, 5-day) |
| **ML Models** | XGBoost (tabular) + LSTM (sequential) |
| **Technical Indicators** | RSI, MACD, Bollinger Bands, SMA crossovers |
| **Backtesting** | 4 strategies vs Buy & Hold benchmark |
| **Dashboard** | Real-time signals, heatmap, model comparison, backtest results |
| **Dynamic Stocks** | Add/remove tickers without touching code |
| **One-Click Update** | Run Full Pipeline button in dashboard |

---

## 📊 Model Performance

Evaluated on out-of-sample test set (975 rows, Jan–Jun 2026):

| Model | Horizon | Accuracy | F1 | AUC-ROC |
|-------|---------|----------|-----|---------|
| XGBoost v2 | T+1d | 0.540 | 0.520 | 0.542 |
| XGBoost v2 | T+3d | 0.520 | 0.474 | 0.567 |
| XGBoost v2 | T+5d | 0.547 | 0.556 | 0.569 |
| **LSTM** | T+1d | **0.550** | **0.583** | **0.560** |
| LSTM | T+3d | 0.533 | 0.521 | 0.554 |
| **LSTM** | T+5d | **0.560** | **0.628** | **0.578** |

> 📌 Financial prediction accuracy of 52–58% is considered strong. All models beat the random 50% baseline. LSTM outperforms XGBoost on T+1 and T+5; XGBoost wins T+3.

---

## 📈 Backtest Results

Tested on Jan–Jun 2026 · $100k initial capital · 0.1% transaction cost per trade:

| Strategy | Return | Sharpe | Max Drawdown | Win Rate |
|----------|--------|--------|-------------|---------|
| 1. Sentiment Long/Short | +288.5% | 11.02 | -5.3% | 84.0% |
| 2. Momentum + Sentiment | +89.7% | 6.50 | -3.5% | 86.0% |
| 3. Mean Reversion | -4.5% | -0.10 | -6.1% | 26.7% |
| 4. SNR + Sentiment | +23.6% | 4.29 | **-0.6%** | **88.7%** |
| Buy & Hold (Benchmark) | +11.0% | 1.31 | -15.0% | — |

> ⚠️ Returns are inflated due to sentiment proxy model (look-ahead bias). SNR + Sentiment shows the best risk-adjusted profile with lowest drawdown.

---

## 🏗️ Project Structure

```
finai-stock-predictor/
├── crawler/
│   └── news_spider.py          # NewsAPI financial news crawler
├── nlp/
│   └── finbert_scorer.py       # FinBERT sentiment analysis
├── dataset/
│   ├── feature_engineering.py  # Technical + sentiment features
│   └── build_historical_dataset.py  # 2-year dataset builder
├── models/
│   ├── train_xgboost_v2.py     # XGBoost trainer
│   └── train_lstm.py           # LSTM trainer
├── strategy/
│   └── indicators.py           # Shared technical indicators
├── backtest/
│   └── run_backtest.py         # 4-strategy backtesting engine
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── data/                       # Generated datasets (gitignored)
├── models/                     # Saved model files (gitignored)
├── config.json                 # Ticker configuration
└── requirements.txt
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.12
- NVIDIA GPU (recommended) — tested on RTX 3060 6GB
- [NewsAPI](https://newsapi.org) free account (100 req/day)

### Installation

```bash
# Clone the repository
git clone https://github.com/PhilipYYW/finai-stock-predictor.git
cd finai-stock-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt

# Set your NewsAPI key
echo NEWS_API_KEY=your_api_key_here > .env
```

### Run the Full Pipeline

```bash
# 1. Fetch news
python crawler/news_spider.py

# 2. Sentiment analysis
python nlp/finbert_scorer.py

# 3. Build dataset
python dataset/build_historical_dataset.py

# 4. Train XGBoost
python models/train_xgboost_v2.py

# 5. Train LSTM
python models/train_lstm.py

# 6. Run backtest
python backtest/run_backtest.py

# 7. Launch dashboard
streamlit run dashboard/app.py
```

Or use the **🔄 Run Full Pipeline** button in the dashboard sidebar.

---

## 🖥️ Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` with 4 pages:

| Page | Content |
|------|---------|
| 📊 Market Overview | Sentiment heatmap + live signals for all stocks |
| 🔍 Stock Deep Dive | Price chart + sentiment + RSI + news + prediction gauge |
| 🤖 Model Performance | XGBoost vs LSTM comparison across horizons |
| 📈 Backtest Results | Strategy returns, Sharpe ratio, max drawdown |

### Adding New Stocks
1. Sidebar → **➕ Add New Stock** → enter ticker + company name
2. Click **🔄 Run Full Pipeline**
3. New stock appears with real predictions in ~5 minutes

---

## 🔬 Technical Details

### Sentiment Features
- Rolling mean sentiment (3-day, 5-day)
- Sentiment momentum (today vs 3-day average)
- Positive/negative ratio over 5 days
- Sentiment volatility (5-day std)

### Technical Indicators
- RSI (14), MACD (12/26/9), Bollinger Bands (20, 2σ)
- SMA5, SMA10, SMA20 crossover signals
- 1-day, 3-day, 5-day price momentum
- 5-day, 20-day volatility
- Volume ratio vs 20-day average

### Trading Strategies
1. **Sentiment Long/Short** — pure model signal baseline
2. **Momentum + Sentiment** — SMA crossover + model confirmation
3. **Mean Reversion** — extreme RSI + sentiment contrarian
4. **SNR + Sentiment** — support/resistance levels + model confirmation

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**. It is not financial advice. Past backtest performance does not guarantee future results. All data sourced from Yahoo Finance and NewsAPI for research use only.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| NLP | FinBERT (HuggingFace Transformers) |
| Deep Learning | PyTorch 2.6 (CUDA) |
| ML | XGBoost, scikit-learn |
| Data | yfinance, NewsAPI, Pandas |
| Dashboard | Streamlit, Plotly |
| Version Control | Git, GitHub |

---

## 📄 License

MIT License — free to use for personal and educational purposes.
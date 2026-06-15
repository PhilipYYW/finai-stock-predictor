#!/bin/bash
echo "============================================"
echo "  FinAI Stock Predictor — Mac/Linux Setup"
echo "============================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.10-3.12"
    exit 1
fi

# Virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# PyTorch (CPU version for Mac/Linux without CUDA)
echo "[2/5] Installing PyTorch..."
pip install torch torchvision torchaudio -q

# Dependencies
echo "[3/5] Installing dependencies..."
pip install -r requirements.txt -q

# API key
echo "[4/5] Checking NewsAPI key..."
if [ ! -f .env ]; then
    echo
    read -p "Enter your NewsAPI key (get free key at newsapi.org): " apikey
    echo "NEWS_API_KEY=$apikey" > .env
    echo "[OK] API key saved"
else
    echo "[OK] .env file found"
fi

# Pipeline
echo "[5/5] Running data pipeline..."
python crawler/news_spider.py
python nlp/finbert_scorer.py
python dataset/build_historical_dataset.py
python models/train_xgboost_v2.py
python backtest/run_backtest.py

echo
echo "============================================"
echo "  Setup complete! Launching dashboard..."
echo "  Open http://localhost:8501 in your browser"
echo "============================================"
streamlit run dashboard/app.py

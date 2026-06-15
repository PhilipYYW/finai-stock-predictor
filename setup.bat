@echo off
echo ============================================
echo   FinAI Stock Predictor — Windows Setup
echo ============================================
echo.

REM Check Python version
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10-3.12 from python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo [1/5] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

REM Install PyTorch (CUDA 12.4)
echo [2/5] Installing PyTorch (GPU version)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

REM Install dependencies
echo [3/5] Installing dependencies...
pip install -r requirements.txt -q

REM Check for .env
echo [4/5] Checking NewsAPI key...
if not exist .env (
    echo.
    echo [ACTION REQUIRED] Enter your NewsAPI key (get free key at newsapi.org):
    set /p apikey="NewsAPI Key: "
    echo NEWS_API_KEY=%apikey% > .env
    echo [OK] API key saved to .env
) else (
    echo [OK] .env file found
)

REM Run pipeline
echo [5/5] Running data pipeline (this takes 5-10 minutes)...
python crawler/news_spider.py
python nlp/finbert_scorer.py
python dataset/build_historical_dataset.py
python models/train_xgboost_v2.py
python backtest/run_backtest.py

echo.
echo ============================================
echo   Setup complete! Launching dashboard...
echo   Open http://localhost:8501 in your browser
echo ============================================
echo.
streamlit run dashboard/app.py

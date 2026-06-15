# backtest/run_backtest.py
"""
Backtests four sentiment-driven trading strategies using
XGBoost v2 predictions on the 2-year historical dataset.

Strategies:
  1. Sentiment Long/Short    — pure model signal
  2. Momentum + Sentiment    — trend filter + model signal
  3. Mean Reversion          — extreme sentiment contrarian
  4. SNR + Sentiment         — support/resistance + model signal

Output: performance table + equity curves saved to data/
"""
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.build_historical_dataset import FEATURE_COLS_HIST, LABEL_COLS
from strategy.indicators import pivot_points, rolling_snr

# ── Configuration ────────────────────────────────────────
INPUT_PATH   = "data/dataset_historical.csv"
OUTPUT_DIR   = "data"
MODEL_DIR    = "models"
LABEL_COL    = "label_1d"          # Primary: T+1
TRANSACTION_COST = 0.001           # 0.1% per trade (realistic taker fee)
INITIAL_CAPITAL  = 100_000         # $100k starting capital
SNR_WINDOW       = 20              # Rolling S/R window
SNR_THRESHOLD    = 0.10            # Within 10% of S/R level = "near level"
# ─────────────────────────────────────────────────────────

FEATURE_COLS = FEATURE_COLS_HIST + ["ticker_enc"]


def load_model_and_scaler(label_col: str):
    model_path  = f"{MODEL_DIR}/xgb_v2_{label_col}.pkl"
    scaler_path = f"{MODEL_DIR}/scaler_v2_{label_col}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def get_predictions(df: pd.DataFrame, model, scaler) -> pd.Series:
    """Get XGBoost probability of UP for each row"""
    X    = scaler.transform(df[FEATURE_COLS].values)
    prob = model.predict_proba(X)[:, 1]
    return pd.Series(prob, index=df.index)


def add_snr_features(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for ticker in df["ticker"].unique():
        t = df[df["ticker"] == ticker].copy().sort_values("date")
        t = rolling_snr(t, window=SNR_WINDOW)
        t = pivot_points(t)
        result.append(t)
    return pd.concat(result).sort_values(["date", "ticker"]).reset_index(drop=True)


# ── Strategy signal generators ────────────────────────────

def signal_long_short(row) -> int:
    """
    Strategy 1: Pure sentiment long/short.
    prob > 0.5 → Long (+1), else Short (-1)
    """
    return 1 if row["prob_up"] > 0.5 else -1


def signal_momentum(row) -> int:
    """
    Strategy 2: Momentum + Sentiment filter.
    Enter only when price trend and model signal agree.
    SMA crossover (sma5 > sma20) = uptrend.
    """
    uptrend   = row.get("sma5_above_sma20", 0) == 1
    prob_up   = row["prob_up"]

    if uptrend and prob_up > 0.55:
        return 1       # Long: trend up + model bullish
    elif not uptrend and prob_up < 0.45:
        return -1      # Short: trend down + model bearish
    return 0           # No trade


def signal_mean_reversion(row) -> int:
    """
    Strategy 3: Mean Reversion on extreme sentiment.
    Extreme negative → buy (expect bounce)
    Extreme positive → sell (expect pullback)
    Uses RSI as additional confirmation.
    """
    sent  = row.get("sentiment_score", 0)
    rsi   = row.get("rsi14", 50)

    if sent < -0.3 and rsi < 35:
        return 1    # Oversold + negative sentiment → contrarian long
    elif sent > 0.3 and rsi > 65:
        return -1   # Overbought + positive sentiment → contrarian short
    return 0


def signal_snr(row) -> int:
    """
    Strategy 4: SNR + Sentiment confirmation.
    Near support + bullish model → long
    Near resistance + bearish model → short
    Filters false breakouts using sentiment.
    """
    dist_sup = row.get("dist_support", 0.5)
    dist_res = row.get("dist_resistance", 0.5)
    prob_up  = row["prob_up"]

    near_support    = dist_sup <= SNR_THRESHOLD
    near_resistance = dist_res <= SNR_THRESHOLD

    if near_support and prob_up > 0.55:
        return 1    # Near support + bullish → long
    elif near_resistance and prob_up < 0.45:
        return -1   # Near resistance + bearish → short
    return 0        # Not near key level → no trade


# ── Backtesting engine ────────────────────────────────────

def backtest_strategy(df: pd.DataFrame,
                      signal_fn,
                      strategy_name: str) -> dict:
    """
    Simple daily rebalancing backtest.
    Each day: compute signal, enter position at close, exit next close.
    Deduct transaction cost on every trade.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    capital   = float(INITIAL_CAPITAL)
    equity    = [capital]
    trades    = 0
    prev_sig  = 0

    for i in range(len(df) - 1):
        row     = df.iloc[i]
        sig     = signal_fn(row)
        ret_1d  = df.iloc[i + 1]["ret_1d"] if "ret_1d" in df.columns else 0.0

        if pd.isna(ret_1d):
            ret_1d = 0.0

        # Position return
        position_ret = sig * ret_1d

        # Transaction cost when signal changes
        if sig != prev_sig and sig != 0:
            position_ret -= TRANSACTION_COST
            trades += 1

        capital *= (1 + position_ret)
        equity.append(capital)
        prev_sig = sig

    equity = np.array(equity)

    # ── Performance metrics ───────────────────────────────
    total_return = (equity[-1] / equity[0] - 1) * 100

    # Daily returns
    daily_ret = np.diff(equity) / equity[:-1]

    # Annualised Sharpe (252 trading days)
    mean_ret = daily_ret.mean()
    std_ret  = daily_ret.std()
    sharpe   = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252)

    # Max drawdown
    peak     = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd   = drawdown.min() * 100

    # Win rate (days with positive return when in position)
    pos_days = daily_ret[daily_ret != 0]
    win_rate = (pos_days > 0).mean() * 100 if len(pos_days) > 0 else 0

    print(f"\n  [{strategy_name}]")
    print(f"  Total Return : {total_return:+.2f}%")
    print(f"  Sharpe Ratio : {sharpe:.4f}")
    print(f"  Max Drawdown : {max_dd:.2f}%")
    print(f"  Win Rate     : {win_rate:.1f}%")
    print(f"  Total Trades : {trades}")

    return {
        "strategy":      strategy_name,
        "total_return":  round(total_return, 2),
        "sharpe":        round(sharpe, 4),
        "max_drawdown":  round(max_dd, 2),
        "win_rate":      round(win_rate, 1),
        "trades":        trades,
        "equity_curve":  equity.tolist(),
    }


def buy_and_hold(df: pd.DataFrame) -> dict:
    """Benchmark: Buy & Hold the equal-weight portfolio"""
    df  = df.copy().sort_values("date").reset_index(drop=True)
    ret = df.groupby("date")["ret_1d"].mean().fillna(0)

    capital = float(INITIAL_CAPITAL)
    equity  = [capital]
    for r in ret:
        capital *= (1 + r)
        equity.append(capital)

    equity       = np.array(equity)
    total_return = (equity[-1] / equity[0] - 1) * 100
    daily_ret    = np.diff(equity) / equity[:-1]
    sharpe       = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * np.sqrt(252)
    peak         = np.maximum.accumulate(equity)
    max_dd       = ((equity - peak) / peak).min() * 100

    print(f"\n  [Buy & Hold (Benchmark)]")
    print(f"  Total Return : {total_return:+.2f}%")
    print(f"  Sharpe Ratio : {sharpe:.4f}")
    print(f"  Max Drawdown : {max_dd:.2f}%")

    return {
        "strategy":     "Buy & Hold",
        "total_return": round(total_return, 2),
        "sharpe":       round(sharpe, 4),
        "max_drawdown": round(max_dd, 2),
        "win_rate":     0,
        "trades":       1,
        "equity_curve": equity.tolist(),
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  Backtesting Engine — 4 Strategies")
    print("=" * 55)

    # Load data
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found.")
        exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"\nLoaded {len(df)} rows")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Add SNR features
    print("Adding S/R levels...")
    df = add_snr_features(df)

    # Load XGBoost model
    print(f"Loading model for {LABEL_COL}...")
    model, scaler = load_model_and_scaler(LABEL_COL)

    # Get predictions
    df["prob_up"] = get_predictions(df, model, scaler)

    # Use test period only (last 20% of dates)
    dates      = sorted(df["date"].unique())
    split_date = dates[int(len(dates) * 0.8)]
    test_df    = df[df["date"] >= split_date].copy()
    print(f"\nBacktest period: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"Trading days   : {test_df['date'].nunique()}")

    # Run all strategies
    print(f"\n{'='*55}")
    print("  Strategy Results")
    print(f"{'='*55}")

    strategies = [
        (signal_long_short,    "1. Sentiment Long/Short"),
        (signal_momentum,      "2. Momentum + Sentiment"),
        (signal_mean_reversion,"3. Mean Reversion"),
        (signal_snr,           "4. SNR + Sentiment"),
    ]

    # Backtest per ticker then average
    all_results = []
    for sig_fn, name in strategies:
        ticker_results = []
        for ticker in test_df["ticker"].unique():
            t_df = test_df[test_df["ticker"] == ticker].copy()
            res  = backtest_strategy(t_df, sig_fn, f"{name} [{ticker}]")
            ticker_results.append(res)

        # Average across tickers
        avg = {
            "strategy":     name,
            "total_return": np.mean([r["total_return"] for r in ticker_results]),
            "sharpe":       np.mean([r["sharpe"]       for r in ticker_results]),
            "max_drawdown": np.mean([r["max_drawdown"] for r in ticker_results]),
            "win_rate":     np.mean([r["win_rate"]     for r in ticker_results]),
            "trades":       int(np.mean([r["trades"]   for r in ticker_results])),
        }
        all_results.append(avg)

    # Buy & Hold benchmark
    bh = buy_and_hold(test_df)
    all_results.append(bh)

    # ── Summary table ─────────────────────────────────────
    print(f"\n{'='*55}")
    print("  Final Backtest Summary (avg across 10 tickers)")
    print(f"{'='*55}")
    print(f"{'Strategy':<30} {'Return':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'WinRate':>9} {'Trades':>7}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['strategy']:<30} {r['total_return']:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['max_drawdown']:>7.2f}% "
              f"{r['win_rate']:>8.1f}% {r['trades']:>7}")

    # Save results
    results_df = pd.DataFrame([{k: v for k, v in r.items()
                                 if k != "equity_curve"}
                                for r in all_results])
    results_df.to_csv(f"{OUTPUT_DIR}/backtest_results.csv",
                      index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_DIR}/backtest_results.csv")
    print("\nDone!")

# strategy/indicators.py
"""
Shared technical indicator utilities for all strategies.
Pivot Points and Rolling High/Low for SNR detection.
"""
import pandas as pd
import numpy as np


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classic Pivot Points using previous day OHLC.
    Adds: pivot, r1, r2, s1, s2
    """
    prev_high  = df["high"].shift(1)
    prev_low   = df["low"].shift(1)
    prev_close = df["close"].shift(1)

    pivot     = (prev_high + prev_low + prev_close) / 3
    df["pivot"] = pivot
    df["r1"]    = 2 * pivot - prev_low
    df["r2"]    = pivot + (prev_high - prev_low)
    df["s1"]    = 2 * pivot - prev_high
    df["s2"]    = pivot - (prev_high - prev_low)
    return df


def rolling_snr(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling Support and Resistance levels.
    Support  = rolling minimum close over window
    Resistance = rolling maximum close over window
    """
    df["support"]    = df["close"].rolling(window).min()
    df["resistance"] = df["close"].rolling(window).max()
    df["snr_range"]  = df["resistance"] - df["support"]

    # Distance from support/resistance as % of range
    df["dist_support"]    = (df["close"] - df["support"]) / (df["snr_range"] + 1e-9)
    df["dist_resistance"] = (df["resistance"] - df["close"]) / (df["snr_range"] + 1e-9)
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

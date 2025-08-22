import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import os

-------------------------------

Utility functions

-------------------------------

def load_tickers(file="tickers.txt"): try: with open(file, "r") as f: return [line.strip() for line in f if line.strip()] except Exception as e: st.error(f"Error loading tickers: {e}") return []

-------------------------------

Indicator functions

-------------------------------

def compute_rsi(series, period=14): delta = series.diff() gain = delta.where(delta > 0, 0.0) loss = -delta.where(delta < 0, 0.0)

avg_gain = gain.rolling(window=period, min_periods=period).mean()
avg_loss = loss.rolling(window=period, min_periods=period).mean()

rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
return rsi

def compute_ema(series, span): return series.ewm(span=span, adjust=False).mean()

def compute_macd(series, fast=12, slow=26, signal=9): ema_fast = compute_ema(series, fast) ema_slow = compute_ema(series, slow) macd = ema_fast - ema_slow signal_line = compute_ema(macd, signal) hist = macd - signal_line return macd, signal_line, hist

def compute_bollinger(series, period=20, num_std=2): sma = series.rolling(window=period).mean() std = series.rolling(window=period).std() upper = sma + (std * num_std) lower = sma - (std * num_std) return sma, upper, lower

def compute_utbot(close, atr_period=10, multiplier=1.0): atr = close.rolling(window=atr_period).std()  # using std as proxy for ATR buy_signal = close > (close.shift(1) + multiplier * atr) sell_signal = close < (close.shift(1) - multiplier * atr) return buy_signal, sell_signal

-------------------------------

Add indicators to DataFrame

-------------------------------

def add_indicators(df): df["RSI"] = compute_rsi(df["Close"]) df["EMA20"] = compute_ema(df["Close"], 20) df["EMA50"] = compute_ema(df["Close"], 50) df["MACD"], df["Signal"], df["Hist"] = compute_macd(df["Close"]) df["SMA20"], df["BB_Upper"], df["BB_Lower"] = compute_bollinger(df["Close"]) df["UT_Buy"], df["UT_Sell"] = compute_utbot(df["Close"]) return df

-------------------------------

Strategy logic

-------------------------------

def generate_signals(df): signals = []

if df["RSI"].iloc[-1] < 30:
    signals.append("RSI Oversold → Possible Buy")
elif df["RSI"].iloc[-1] > 70:
    signals.append("RSI Overbought → Possible Sell")

if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]:
    signals.append("EMA20 > EMA50 → Bullish")
else:
    signals.append("EMA20 < EMA50 → Bearish")

if df["MACD"].iloc[-1] > df["Signal"].iloc[-1]:
    signals.append("MACD Crossover → Bullish")
else:
    signals.append("MACD Crossover → Bearish")

if df["Close"].iloc[-1] > df["BB_Upper"].iloc[-1]:
    signals.append("Price above Bollinger Upper → Overbought")
elif df["Close"].iloc[-1] < df["BB_Lower"].iloc[-1]:
    signals.append("Price below Bollinger Lower → Oversold")

if df["UT_Buy"].iloc[-1]:
    signals.append("UT Bot Buy Signal")
if df["UT_Sell"].iloc[-1]:
    signals.append("UT Bot Sell Signal")

return signals

-------------------------------

Streamlit App

-------------------------------

def main(): st.set_page_config(page_title="Stock Screener", layout="wide") st.title("📈 Stock Screener with Indicators")

tickers = load_tickers()
selected = st.selectbox("Select a Ticker", tickers)

if selected:
    end = dt.date.today()
    start = end - dt.timedelta(days=365)

    df = yf.download(selected, start=start, end=end)

    if df.empty:
        st.error("No data found for this ticker.")
        return

    df = add_indicators(df)

    # Show latest signals
    st.subheader(f"Latest Signals for {selected}")
    signals = generate_signals(df)
    for s in signals:
        st.write("-", s)

    # Show latest data
    st.subheader("Recent Data with Indicators")
    st.dataframe(df.tail(20))

if name == "main": main()


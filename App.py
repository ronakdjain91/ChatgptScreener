import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------
# Utility functions
# -------------------------------

def load_tickers(file="tickers.txt"):
    """Loads tickers from a file."""
    try:
        with open(file, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        return []

# -------------------------------
# Indicator functions
# -------------------------------

def compute_rsi(series, period=14):
    """Computes the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_ema(series, span):
    """Computes the Exponential Moving Average (EMA)."""
    return series.ewm(span=span, adjust=False).mean()

def compute_macd(series, fast=12, slow=26, signal=9):
    """Computes the Moving Average Convergence Divergence (MACD)."""
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = compute_ema(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_bollinger(series, period=20, num_std=2):
    """Computes the Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return sma, upper, lower

def compute_atr(high, low, close, period=14):
    """Computes the Average True Range (ATR)."""
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def compute_utbot(close, high, low, atr_period=10, multiplier=1.0):
    """Computes the UT Bot signals."""
    atr = compute_atr(high, low, close, period=atr_period)
    buy_signal = close > (close.shift(1) + multiplier * atr)
    sell_signal = close < (close.shift(1) - multiplier * atr)
    return buy_signal, sell_signal

# -------------------------------
# Add indicators to DataFrame
# -------------------------------

def add_indicators(df, rsi_period, ema_fast, ema_slow, macd_signal, bb_period, bb_std, ut_atr_period, ut_multiplier):
    """Adds technical indicators to the DataFrame."""
    df["RSI"] = compute_rsi(df["Close"], period=rsi_period)
    df["EMA20"] = compute_ema(df["Close"], ema_fast)
    df["EMA50"] = compute_ema(df["Close"], ema_slow)
    df["MACD"], df["Signal"], df["Hist"] = compute_macd(df["Close"], fast=ema_fast, slow=ema_slow, signal=macd_signal)
    df["SMA20"], df["BB_Upper"], df["BB_Lower"] = compute_bollinger(df["Close"], period=bb_period, num_std=bb_std)
    df["UT_Buy"], df["UT_Sell"] = compute_utbot(df["Close"], df["High"], df["Low"], atr_period=ut_atr_period, multiplier=ut_multiplier)
    return df

# -------------------------------
# Strategy logic
# -------------------------------

def generate_signals(df):
    """Generates trading signals based on the latest indicator values."""
    signals = []

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

def plot_data(df, ticker):
    """Plots the stock data and indicators."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=(f'{ticker} Price', 'RSI', 'MACD'),
                        row_heights=[0.6, 0.2, 0.2])

    # Price chart
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'], name='Price'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20', line=dict(color='orange', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50', line=dict(color='purple', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dash'), fill='tonexty'),
                  row=1, col=1)

    # RSI chart
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=2, col=1)
    fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="red", opacity=0.2, row=2, col=1)

    # MACD chart
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='orange', width=1)),
                  row=3, col=1)
    fig.add_bar(x=df.index, y=df['Hist'], name='Histogram', marker_color=np.where(df['Hist'] > 0, 'green', 'red')),
                  row=3, col=1)

    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Streamlit App
# -------------------------------

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Stock Screener", layout="wide")
    st.title("📈 Stock Screener with Indicators")

    # Sidebar for indicator settings
    st.sidebar.header("Indicator Settings")
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    ema_fast = st.sidebar.slider("EMA Fast", 5, 50, 12)
    ema_slow = st.sidebar.slider("EMA Slow", 10, 100, 26)
    macd_signal = st.sidebar.slider("MACD Signal", 5, 20, 9)
    bb_period = st.sidebar.slider("Bollinger Band Period", 10, 50, 20)
    bb_std = st.sidebar.slider("Bollinger Band Std Dev", 1.0, 3.0, 2.0, 0.5)
    ut_atr_period = st.sidebar.slider("UT Bot ATR Period", 5, 20, 10)
    ut_multiplier = st.sidebar.slider("UT Bot Multiplier", 0.5, 3.0, 1.0, 0.1)

    # Ticker selection
    tickers = load_tickers()
    selected = st.selectbox("Select a Ticker", tickers)

    if selected:
        # Download data
        end = dt.date.today()
        start = end - dt.timedelta(days=365)

        try:
            df = yf.download(selected, start=start, end=end)
            if df.empty:
                st.error("No data found for this ticker. It may be delisted or the ticker symbol is incorrect.")
                return
        except Exception as e:
            st.error(f"Error downloading data for {selected}: {e}")
            return

        # Calculate indicators
        df = add_indicators(df, rsi_period, ema_fast, ema_slow, macd_signal, bb_period, bb_std, ut_atr_period, ut_multiplier)

        # Display signals and charts
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader(f"Signals for {selected}")
            signals = generate_signals(df)
            for s in signals:
                st.write(s)

        with col2:
            st.subheader("Charts")
            plot_data(df, selected)

        # Display recent data
        st.subheader("Recent Data")
        st.dataframe(df.tail(10))


if __name__ == "__main__":
    main()


# App.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

# -------------------------------
# Load Tickers
# -------------------------------
@st.cache_data
def load_tickers():
    try:
        with open("tickers.txt", "r") as f:
            tickers = [line.strip() for line in f.readlines()]
        return tickers
    except:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]

TICKERS = load_tickers()

# -------------------------------
# Fetch Stock Data
# -------------------------------
@st.cache_data
def get_stock_data(ticker, years=5):
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    df = yf.download(ticker, start=start, end=end, interval="1wk")
    df.dropna(inplace=True)
    return df

# -------------------------------
# Simple Technical Indicators
# -------------------------------
def add_indicators(df):
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -------------------------------
# Strategy Presets
# -------------------------------
STRATEGIES = {
    "Value": {"PE": (0, 15), "ROE": (15, 100), "DebtEq": (0, 1)},
    "Growth": {"RevenueGrowth": (10, 100), "EarningsGrowth": (15, 100)},
    "Momentum": {"RSI": (50, 80), "EMA200": "above"},
    "CAN SLIM": {"EarningsGrowth": (15, 100), "RevenueGrowth": (10, 100), "RSI": (40, 80)},
    "Quality": {"ROE": (20, 100), "DebtEq": (0, 0.5)}
}

# -------------------------------
# Mock Fundamentals
# (In practice, connect to a fundamentals API)
# -------------------------------
def get_mock_fundamentals(ticker):
    np.random.seed(abs(hash(ticker)) % (10**6))  # stable mock data
    return {
        "PE": np.random.uniform(5, 40),
        "ROE": np.random.uniform(5, 30),
        "DebtEq": np.random.uniform(0, 2),
        "RevenueGrowth": np.random.uniform(-5, 30),
        "EarningsGrowth": np.random.uniform(-10, 40),
    }

# -------------------------------
# Screening Function
# -------------------------------
def screen_stocks(strategy_name):
    filters = STRATEGIES[strategy_name]
    results = []
    for t in TICKERS:
        fundamentals = get_mock_fundamentals(t)
        passed = True
        for key, val in filters.items():
            if key == "EMA200":
                df = get_stock_data(t, years=1)
                df = add_indicators(df)
                if not (df["Close"].iloc[-1] > df["EMA200"].iloc[-1]):
                    passed = False
                    break
            else:
                low, high = val
                if not (low <= fundamentals.get(key, 0) <= high):
                    passed = False
                    break
        if passed:
            fundamentals["Ticker"] = t
            results.append(fundamentals)
    return pd.DataFrame(results)

# -------------------------------
# Portfolio (Paper Trading)
# -------------------------------
class Portfolio:
    def __init__(self, cash=100000):
        self.cash = cash
        self.positions = {}

    def buy(self, ticker, price, qty):
        cost = price * qty
        if self.cash >= cost:
            self.cash -= cost
            self.positions[ticker] = self.positions.get(ticker, 0) + qty
            return True
        return False

    def sell(self, ticker, price, qty):
        if ticker in self.positions and self.positions[ticker] >= qty:
            self.positions[ticker] -= qty
            self.cash += price * qty
            return True
        return False

    def value(self, prices):
        val = self.cash
        for t, qty in self.positions.items():
            val += qty * prices.get(t, 0)
        return val

portfolio = Portfolio()

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="📈 Stock Screener & Paper Trading", layout="wide")

st.title("📈 Stock Analysis & Paper Trading App")
st.write("Analyze stocks with 5-year weekly data and strategy presets. Trade virtually with ₹100,000 starting capital.")

tab1, tab2, tab3 = st.tabs(["🔍 Screening", "📊 Visual Insights", "💼 Portfolio"])

# -------------------------------
# Screening Tab
# -------------------------------
with tab1:
    strategy = st.selectbox("Choose Strategy Preset", list(STRATEGIES.keys()))
    df_screened = screen_stocks(strategy)

    if not df_screened.empty:
        st.subheader(f"Stocks matching **{strategy} strategy**")
        st.dataframe(df_screened)

        selected = st.selectbox("Select Ticker for Chart", df_screened["Ticker"].tolist())
        if selected:
            df = get_stock_data(selected)
            df = add_indicators(df)
            fig = px.line(df, x=df.index, y=["Close", "EMA200"], title=f"{selected} Price & EMA200")
            st.plotly_chart(fig, use_container_width=True)

            price = df["Close"].iloc[-1]
            st.write(f"Latest Price: ₹{price:.2f}")
            qty = st.number_input("Quantity", 1, 100, 10)
            if st.button("Buy"):
                if portfolio.buy(selected, price, qty):
                    st.success(f"Bought {qty} shares of {selected} at ₹{price:.2f}")
                else:
                    st.error("Not enough cash.")
            if st.button("Sell"):
                if portfolio.sell(selected, price, qty):
                    st.success(f"Sold {qty} shares of {selected} at ₹{price:.2f}")
                else:
                    st.error("Not enough holdings.")
    else:
        st.warning("No stocks matched this strategy.")

# -------------------------------
# Visual Insights Tab
# -------------------------------
with tab2:
    st.subheader("Sector-wise Averages (Mock Data)")
    sector_data = pd.DataFrame({
        "Sector": ["Tech", "Finance", "Energy", "Auto"],
        "Avg Fundamental Score": np.random.uniform(50, 80, 4),
        "Avg Technical Score": np.random.uniform(40, 90, 4),
    })
    fig1 = px.bar(sector_data, x="Sector", y="Avg Fundamental Score", title="Average Fundamental Score by Sector")
    fig2 = px.bar(sector_data, x="Sector", y="Avg Technical Score", title="Average Technical Score by Sector")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Portfolio Tab
# -------------------------------
with tab3:
    prices = {t: get_stock_data(t, years=1)["Close"].iloc[-1] for t in portfolio.positions.keys()} if portfolio.positions else {}
    st.metric("Cash", f"₹{portfolio.cash:,.2f}")
    st.metric("Total Portfolio Value", f"₹{portfolio.value(prices):,.2f}")
    if portfolio.positions:
        st.subheader("Current Holdings")
        st.write(portfolio.positions)
    else:
        st.info("No holdings yet. Buy stocks from the Screening tab.")

import dash
from dash import dcc, html, Input, Output
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
        print(f"Error loading tickers: {e}")
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

def calculate_indicators(df):
    """Calculates all technical indicators for a given DataFrame."""
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['Signal'], _ = compute_macd(df['Close'])

    # Volume Buzz
    df['AvgVol50'] = df['Volume'].rolling(window=50).mean()
    df['VolumeBuzz'] = (df['Volume'] / df['AvgVol50']).round(2)

    # 52-week High/Low
    df['52WkHigh'] = df['Close'].rolling(window=252).max()
    df['52WkLow'] = df['Close'].rolling(window=252).min()
    df['PctFrom52WkHigh'] = ((df['Close'] - df['52WkHigh']) / df['52WkHigh'] * 100).round(2)

    return df

def run_screener_engine():
    """
    Downloads data for all tickers, calculates indicators,
    and returns both a summary DataFrame and a dict of detailed DataFrames.
    """
    tickers = load_tickers()
    if not tickers:
        return pd.DataFrame(), {}

    # Add market index to the list
    market_index = '^NSEI' # Nifty 50 for India
    all_symbols = tickers + [market_index]

    # Download data for all symbols at once
    df_all = yf.download(all_symbols, period='1y', group_by='ticker')

    # Store detailed dataframes and summary results
    detailed_dfs = {}
    summary_results = []

    for ticker in tickers:
        try:
            df_ticker = df_all[ticker].copy().dropna()
            if df_ticker.empty:
                continue

            df_ticker = calculate_indicators(df_ticker)
            detailed_dfs[ticker] = df_ticker.reset_index() # Store full data for detail view

            latest = df_ticker.iloc[-1]

            # Signals
            golden_cross = latest['MA50'] > latest['MA200'] and df_ticker['MA50'].iloc[-2] < df_ticker['MA200'].iloc[-2]
            macd_cross = latest['MACD'] > latest['Signal'] and df_ticker['MACD'].iloc[-2] < df_ticker['Signal'].iloc[-2]

            summary_results.append({
                'Ticker': ticker,
                'Price': latest['Close'],
                'RSI': latest['RSI'],
                'VolumeBuzz': latest['VolumeBuzz'],
                '% from 52Wk High': latest['PctFrom52WkHigh'],
                'GoldenCross': 'Yes' if golden_cross else 'No',
                'MACDCross': 'Yes' if macd_cross else 'No',
            })
        except (KeyError, IndexError):
            print(f"Could not process data for {ticker}. It might be delisted or have insufficient data.")
            continue

    summary_df = pd.DataFrame(summary_results).round(2)

    # Check market trend
    if market_index in df_all:
        df_market = df_all[market_index].copy().dropna()
        if not df_market.empty:
            market_ma50 = df_market['Close'].rolling(window=50).mean().iloc[-1]
            market_is_up = df_market['Close'].iloc[-1] > market_ma50
            summary_df['MarketTrend'] = 'Up' if market_is_up else 'Down'

    return summary_df, detailed_dfs

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

from dash import dash_table

from flask_caching import Cache

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Setup caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})
TIMEOUT = 3600 # Cache timeout in seconds (1 hour)

@cache.memoize(timeout=TIMEOUT)
def get_screener_data():
    return run_screener_engine()

# Main layout
app.layout = html.Div([
    html.H1("Stock Screener", style={'textAlign': 'center'}),

    html.Div([
        html.H3("Filter Controls"),
        html.Div([
            html.Div([
                html.Label("Signal Crosses:"),
                dcc.Checklist(
                    id='cross-checklist',
                    options=[
                        {'label': 'Golden Cross', 'value': 'GoldenCross'},
                        {'label': 'MACD Crossover', 'value': 'MACDCross'},
                    ],
                    value=[],
                    inline=True
                ),
            ], className='control-item'),

            html.Div([
                html.Label("Market Trend:"),
                dcc.Dropdown(
                    id='market-trend-dropdown',
                    options=[
                        {'label': 'Any', 'value': 'Any'},
                        {'label': 'Up', 'value': 'Up'},
                        {'label': 'Down', 'value': 'Down'}
                    ],
                    value='Any'
                ),
            ], className='control-item'),

            html.Div([
                html.Label("RSI Range:"),
                dcc.RangeSlider(
                    id='rsi-slider',
                    min=0, max=100, step=1, value=[0, 100],
                    marks={i: str(i) for i in range(0, 101, 10)}
                ),
            ], className='control-item'),

            html.Div([
                html.Label("Min Volume Buzz:"),
                dcc.Input(id='vol-buzz-input', type='number', value=0),
            ], className='control-item'),

            html.Div([
                html.Label("Max % from 52Wk High:"),
                dcc.Input(id='high-pct-input', type='number', value=0),
            ], className='control-item'),
        ], className='controls-container'),
    ]),

    dcc.Loading(
        id="loading-screener",
        type="default",
        children=html.Div(id="screener-output")
    ),

    html.Hr(),
    html.H3("Detailed Data View"),
    dcc.Loading(
        id="loading-detail",
        type="default",
        children=html.Div(id="detail-output")
    ),

], style={'padding': '10px'})

# -------------------------------
# Callbacks
# -------------------------------
@app.callback(
    Output('screener-output', 'children'),
    [Input('cross-checklist', 'value'),
     Input('market-trend-dropdown', 'value'),
     Input('rsi-slider', 'value'),
     Input('vol-buzz-input', 'value'),
     Input('high-pct-input', 'value')]
)
def update_screener_table(crosses, market_trend, rsi_range, min_buzz, max_pct_from_high):
    summary_df, _ = get_screener_data() # Unpack the tuple, ignore detailed data for now

    if summary_df.empty:
        return html.P("Could not load screener data.")

    # Apply filters
    filtered_df = summary_df.copy()

    if 'GoldenCross' in crosses:
        filtered_df = filtered_df[filtered_df['GoldenCross'] == 'Yes']
    if 'MACDCross' in crosses:
        filtered_df = filtered_df[filtered_df['MACDCross'] == 'Yes']

    if market_trend != 'Any':
        filtered_df = filtered_df[filtered_df['MarketTrend'] == market_trend]

    filtered_df = filtered_df[
        (filtered_df['RSI'] >= rsi_range[0]) & (filtered_df['RSI'] <= rsi_range[1])
    ]

    if min_buzz is not None and min_buzz > 0:
        filtered_df = filtered_df[filtered_df['VolumeBuzz'] >= min_buzz]

    if max_pct_from_high is not None and max_pct_from_high <= 0:
        filtered_df = filtered_df[filtered_df['% from 52Wk High'] >= max_pct_from_high]

    if filtered_df.empty:
        return html.P("No stocks match the selected criteria.")

    return dash_table.DataTable(
        id='screener-table',
        columns=[{"name": i, "id": i} for i in filtered_df.columns],
        data=filtered_df.to_dict('records'),
        sort_action="native",
        filter_action="native",
        page_size=20,
        style_cell={'textAlign': 'left'},
        style_header={'fontWeight': 'bold'},
    )

@app.callback(
    Output('detail-output', 'children'),
    [Input('screener-table', 'active_cell'),
     Input('screener-table', 'data')] # Get the filtered data from the table
)
def update_detail_view(active_cell, table_data):
    if not active_cell or not table_data:
        return html.P("Click on a stock in the table above to see its detailed data.")

    # Get the ticker from the clicked row
    row_index = active_cell['row']
    clicked_ticker = table_data[row_index]['Ticker']

    # Get the detailed data from the cache
    _, detailed_dfs = get_screener_data()

    if clicked_ticker not in detailed_dfs:
        return html.P(f"No detailed data available for {clicked_ticker}.")

    detail_df = detailed_dfs[clicked_ticker]
    detail_df['Date'] = pd.to_datetime(detail_df['Date']).dt.strftime('%Y-%m-%d')

    return dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in detail_df.columns],
        data=detail_df.to_dict('records'),
        page_size=15,
        style_table={'overflowX': 'auto'},
    )

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)

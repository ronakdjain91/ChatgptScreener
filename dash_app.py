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

from dash import dash_table

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Load tickers for dropdown
tickers = load_tickers()
ticker_options = [{'label': ticker, 'value': ticker} for ticker in tickers]

# Define the app layout
app.layout = html.Div([
    html.H1("📈 Dash Stock Screener with Indicators", style={'textAlign': 'center'}),

    # Main container
    html.Div([
        # Sidebar for controls
        html.Div([
            html.H2("Controls"),
            html.Label("Select Ticker:"),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=ticker_options,
                value=tickers[0] if tickers else None
            ),
            html.Hr(),
            html.H3("Indicator Settings"),

            html.Label("RSI Period:"),
            dcc.Slider(id='rsi-slider', min=5, max=30, step=1, value=14, marks={i: str(i) for i in range(5, 31, 5)}),

            html.Label("EMA Fast:"),
            dcc.Slider(id='ema-fast-slider', min=5, max=50, step=1, value=12, marks={i: str(i) for i in range(5, 51, 5)}),

            html.Label("EMA Slow:"),
            dcc.Slider(id='ema-slow-slider', min=10, max=100, step=1, value=26, marks={i: str(i) for i in range(10, 101, 10)}),

            html.Label("MACD Signal:"),
            dcc.Slider(id='macd-signal-slider', min=5, max=20, step=1, value=9, marks={i: str(i) for i in range(5, 21, 5)}),

            html.Label("Bollinger Period:"),
            dcc.Slider(id='bb-period-slider', min=10, max=50, step=1, value=20, marks={i: str(i) for i in range(10, 51, 10)}),

            html.Label("Bollinger Std Dev:"),
            dcc.Slider(id='bb-std-slider', min=1, max=3, step=0.5, value=2, marks={i: str(i) for i in [1, 1.5, 2, 2.5, 3]}),

            html.Label("UT Bot ATR Period:"),
            dcc.Slider(id='ut-atr-period-slider', min=5, max=20, step=1, value=10, marks={i: str(i) for i in range(5, 21, 5)}),

            html.Label("UT Bot Multiplier:"),
            dcc.Slider(id='ut-multiplier-slider', min=0.5, max=3, step=0.1, value=1.0, marks={i: str(i) for i in [0.5, 1, 1.5, 2, 2.5, 3]}),

        ], style={'width': '25%', 'float': 'left', 'padding': '10px', 'boxSizing': 'border-box'}),

        # Main content area
        html.Div([
            html.Div([
                html.H3(id='signals-header'),
                html.Div(id='signals-output')
            ]),
            dcc.Graph(id='main-chart'),
            html.H3("Recent Data"),
            dash_table.DataTable(
                id='data-table',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            )
        ], style={'width': '75%', 'float': 'right', 'padding': '10px', 'boxSizing': 'border-box'})
    ], style={'display': 'flex', 'flexDirection': 'row'})
])


# -------------------------------
# Charting function
# -------------------------------
def create_figure(df, ticker):
    """Creates the Plotly figure with all charts."""
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
    fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='Histogram', marker_color=np.where(df['Hist'] > 0, 'green', 'red')),
                  row=3, col=1)

    fig.update_layout(height=800, showlegend=False, xaxis_rangeslider_visible=False)
    return fig

# -------------------------------
# Callbacks
# -------------------------------
@app.callback(
    [Output('main-chart', 'figure'),
     Output('signals-header', 'children'),
     Output('signals-output', 'children'),
     Output('data-table', 'data'),
     Output('data-table', 'columns')],
    [Input('ticker-dropdown', 'value'),
     Input('rsi-slider', 'value'),
     Input('ema-fast-slider', 'value'),
     Input('ema-slow-slider', 'value'),
     Input('macd-signal-slider', 'value'),
     Input('bb-period-slider', 'value'),
     Input('bb-std-slider', 'value'),
     Input('ut-atr-period-slider', 'value'),
     Input('ut-multiplier-slider', 'value')]
)
def update_dashboard(ticker, rsi_period, ema_fast, ema_slow, macd_signal, bb_period, bb_std, ut_atr_period, ut_multiplier):
    if not ticker:
        return go.Figure(), "Select a ticker", [], [], []

    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=365)
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            return go.Figure(), f"No data for {ticker}", [html.P("Could not retrieve data.")], [], []
    except Exception as e:
        return go.Figure(), f"Error loading {ticker}", [html.P(str(e))], [], []

    df = add_indicators(df, rsi_period, ema_fast, ema_slow, macd_signal, bb_period, bb_std, ut_atr_period, ut_multiplier)

    # Create figure
    fig = create_figure(df.tail(200), ticker) # Plot last 200 days for clarity

    # Generate signals
    signals = generate_signals(df)
    signals_header = f"Signals for {ticker}"
    signals_output = [html.P(s) for s in signals]

    # Prepare data table
    df_display = df.tail(10).round(2).reset_index()
    df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d')
    table_data = df_display.to_dict('records')
    table_columns = [{"name": i, "id": i} for i in df_display.columns]

    return fig, signals_header, signals_output, table_data, table_columns

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

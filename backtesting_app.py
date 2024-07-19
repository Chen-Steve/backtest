import yfinance as yf
import pandas as pd
import streamlit as st
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, SignalStrategy
from backtesting.test import SMA
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import report_generator

# Define Bollinger Bands as a standalone function
def BollingerBands(series, period, std_dev):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band

# Define a strategy with manually calculated RSI and Bollinger Bands indicators
class OptimizableStrategy(SignalStrategy):
    n1 = 10
    n2 = 20
    rsi_period = 14
    bb_period = 20
    bb_std_dev = 2

    def init(self):
        super().init()
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
        self.rsi = self.I(self.RSI, pd.Series(self.data.Close), self.rsi_period)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(BollingerBands, pd.Series(self.data.Close), self.bb_period, self.bb_std_dev)

    def RSI(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def next(self):
        super().next()
        if crossover(self.sma1, self.sma2) and self.rsi[-1] < 30 and self.data.Close[-1] < self.bb_lower[-1]:
            self.buy()
        elif crossover(self.sma2, self.sma1) or self.rsi[-1] > 70 or self.data.Close[-1] > self.bb_upper[-1]:
            self.sell()

# Streamlit UI
st.title('Yahoo Finance Backtest')
st.sidebar.header('User Input Parameters')

# Sidebar - User Input
ticker = st.sidebar.text_input('Ticker', 'AAPL')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))
short_window = st.sidebar.slider('Short Window', min_value=1, max_value=100, value=10)
long_window = st.sidebar.slider('Long Window', min_value=1, max_value=300, value=20)
rsi_period = st.sidebar.slider('RSI Period', min_value=1, max_value=50, value=14)
bb_period = st.sidebar.slider('Bollinger Bands Period', min_value=1, max_value=50, value=20)
bb_std_dev = st.sidebar.slider('Bollinger Bands Std Dev', min_value=1, max_value=5, value=2)

# Function to download data
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    return data

# Get Data
data = get_data(ticker, start_date, end_date)

# Update Strategy parameters
OptimizableStrategy.n1 = short_window
OptimizableStrategy.n2 = long_window
OptimizableStrategy.rsi_period = rsi_period
OptimizableStrategy.bb_period = bb_period
OptimizableStrategy.bb_std_dev = bb_std_dev

# Run Backtest with Optimization
bt = Backtest(data, OptimizableStrategy, cash=10000, commission=.002)
stats = bt.optimize(n1=range(5, 20, 1), n2=range(20, 50, 1), maximize='Equity Final [$]')

# Compute the equity curve manually from the trades data
def compute_equity_curve(output):
    trades = output['_trades']
    initial_cash = 10000
    equity = initial_cash
    equity_curve = []
    for index, row in trades.iterrows():
        if row['Size'] > 0:  # Buy
            equity -= row['EntryPrice'] * row['Size']
        else:  # Sell
            equity += row['EntryPrice'] * abs(row['Size'])
        equity_curve.append(equity)
    return equity_curve

# Plot results using Plotly
def plot_backtest_results(output, data):
    trades = output['_trades']
    
    data['SMA1'] = data['Close'].rolling(window=OptimizableStrategy.n1).mean()
    data['SMA2'] = data['Close'].rolling(window=OptimizableStrategy.n2).mean()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['BB Upper'], data['BB Middle'], data['BB Lower'] = BollingerBands(data['Close'], bb_period, bb_std_dev)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        subplot_titles=('Price and SMA', 'RSI', 'Bollinger Bands', 'Equity Curve'),
                        vertical_spacing=0.1)

    # Plotting the close price and SMAs
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA1'], name=f'SMA{OptimizableStrategy.n1}', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA2'], name=f'SMA{OptimizableStrategy.n2}', line=dict(color='red')), row=1, col=1)

    # Plotting buy and sell signals
    buys = trades.loc[trades['Size'] > 0]
    sells = trades.loc[trades['Size'] < 0]
    fig.add_trace(go.Scatter(x=buys['EntryTime'], y=buys['EntryPrice'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells['EntryTime'], y=sells['EntryPrice'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal'), row=1, col=1)

    # Plotting RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='orange')), row=2, col=1)
    fig.add_shape(dict(type="line", x0=data.index[0], y0=30, x1=data.index[-1], y1=30, line=dict(color="green", width=2, dash="dash")), row=2, col=1)
    fig.add_shape(dict(type="line", x0=data.index[0], y0=70, x1=data.index[-1], y1=70, line=dict(color="red", width=2, dash="dash")), row=2, col=1)

    # Plotting Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['BB Upper'], name='BB Upper', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB Middle'], name='BB Middle', line=dict(color='black')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB Lower'], name='BB Lower', line=dict(color='red')), row=3, col=1)

    # Plotting Equity Curve
    equity_curve = compute_equity_curve(output)
    fig.add_trace(go.Scatter(x=trades['EntryTime'], y=equity_curve, name='Equity Curve', line=dict(color='purple')), row=4, col=1)

    fig.update_layout(title='Backtest Results', showlegend=True, height=1200)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='RSI', row=2, col=1)
    fig.update_yaxes(title_text='Bollinger Bands', row=3, col=1)
    fig.update_yaxes(title_text='Equity', row=4, col=1)

    return fig

# Plot the backtest results
fig = plot_backtest_results(stats, data)

# Display results
st.plotly_chart(fig, use_container_width=True)
st.write(stats)

# Calculate and display performance metrics
def calculate_performance_metrics(output):
    performance_metrics = {
        'Total Return': output['Return [%]'],
        'Annualized Return': output['Return (Ann.) [%]'],
        'Sharpe Ratio': output['Sharpe Ratio'],
        'Max Drawdown': output['Max. Drawdown [%]'],
    }
    return performance_metrics

metrics = calculate_performance_metrics(stats)
st.write("Performance Metrics:")
st.write(metrics)

# Generate and download report
if st.button('Generate Report'):
    report = report_generator.generate_report(stats, metrics, data, stats['_trades'])
    buffer = report_generator.download_report(report)
    st.download_button(label='Download Report', data=buffer, file_name='backtest_report.txt', mime='text/plain')
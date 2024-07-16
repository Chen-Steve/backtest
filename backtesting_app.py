import yfinance as yf
import pandas as pd
import streamlit as st
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import matplotlib.pyplot as plt
from io import BytesIO

# Define a strategy
class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

# Streamlit UI
st.title('Backtesting with backtesting.py')
st.sidebar.header('User Input Parameters')

# Sidebar - User Input
ticker = st.sidebar.text_input('Ticker', 'AAPL')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))
short_window = st.sidebar.slider('Short Window', min_value=1, max_value=100, value=10)
long_window = st.sidebar.slider('Long Window', min_value=1, max_value=300, value=20)

# Function to download data
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Get Data
data = get_data(ticker, start_date, end_date)

# Update Strategy parameters
SmaCross.n1 = short_window
SmaCross.n2 = long_window

# Run Backtest
bt = Backtest(data, SmaCross, cash=10000, commission=.002)
output = bt.run()

# Plot results using matplotlib
def plot_backtest_results(output, data):
    trades = output['_trades']
    
    data['SMA1'] = data['Close'].rolling(window=SmaCross.n1).mean()
    data['SMA2'] = data['Close'].rolling(window=SmaCross.n2).mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Price', color='black')
    ax.plot(data.index, data['SMA1'], label=f'SMA{SmaCross.n1}', color='blue')
    ax.plot(data.index, data['SMA2'], label=f'SMA{SmaCross.n2}', color='red')
    
    # Plot buy signals
    buys = trades.loc[trades['Size'] > 0]
    ax.scatter(buys['EntryTime'], buys['EntryPrice'], marker='^', color='green', label='Buy Signal')

    # Plot sell signals
    sells = trades.loc[trades['Size'] < 0]
    ax.scatter(sells['ExitTime'], sells['ExitPrice'], marker='v', color='red', label='Sell Signal')

    ax.set_title('Backtest Results')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    
    return fig

# Plot the backtest results
fig = plot_backtest_results(output, data)

# Save plot to a BytesIO object
buf = BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)

# Display results
st.write(output)
st.image(buf, width=800)

import sys
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from prophet import Prophet
from pandas.plotting import autocorrelation_plot
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import pandas_datareader as pdr
import pytictoc as tt
from tickerTA import Ticker
from tickerTA import TechnicalAnalysis
from tradingStrategy1 import TradingStrategy1


def get_tickers500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')

    table = soup.find('table', {'id': 'constituents'})

    tickers500 = []
    for row in table.find_all('tr')[1:]:
        ticker = row.find('td').text.strip()
        tickers500.append(ticker)

    return tickers500



def add_z_score(data: pd.DataFrame, frame: str = 'Adj Close', window1 = 20) -> pd.DataFrame:
    data['Z Score ' + frame] = (data[frame] - data[frame].rolling(window= window1).mean()) / data[frame].rolling(window= window1).std()
    data.dropna(inplace=True)
    return data

def get_z_score(data: pd.DataFrame, frame: str = 'Adj Close', window1 = 20) -> pd.DataFrame:
    data['Z Score ' + frame] = (data[frame] - data[frame].rolling(window= window1).mean()) / data[frame].rolling(window= window1).std()
    data.dropna(inplace=True)
    return data['Z Score ' + frame]

def generate_returns(dataframe: pd.DataFrame):
    returns_df = dataframe.copy(deep=True)
    returns_df['Returns'] = returns_df['Sell Price'] - returns_df['Buy Price']
    returns_df['Returns %'] = (returns_df['Returns'] / returns_df['Buy Price']) * 100
    returns_df['Cumulative Returns %'] = returns_df['Returns %'].cumsum()
    returns_df['Cumulative Returns %'] = returns_df['Cumulative Returns %'].shift(1)
    returns_df['Cumulative Returns %'].iloc[0] = 0
    returns_df['Cumulative Returns %'] = returns_df['Cumulative Returns %'].fillna(0)
    returns_df['Cumulative Returns %'] = returns_df['Cumulative Returns %'].round(2)
    returns_df['Profitable'] = returns_df['Returns'] > 0
    returns_df['Profitable'] = returns_df['Profitable'].replace({True: 'Yes', False: 'No'})
    return returns_df



stock_df = pd.DataFrame()
tradingstrategy1_df = pd.DataFrame()

for ticker in tickers500[1:500]:
    stock = Ticker(ticker, start='2022-01-31', end='2024-01-31')
    stock_df = stock_df._append(stock.df)
    techA = TechnicalAnalysis(stock)
    tradingS = TradingStrategy1(techA)
    tradingstrategy1_df = tradingstrategy1_df._append(tradingS.trades_df)

# stock_df.reset_index(drop=True, inplace=True)
stock_df.to_csv('BackTestData/stock_df.csv')

tradingstrategy1_df.sort_values(by=['Sell Date'], ascending=[True], inplace=True)
tradingstrategy1_df.reset_index(drop=True, inplace=True)
tradingstrategy1_df.to_csv('tradingstrategy1.csv', index=False)

tradingstrategy1_df['Profitable'].value_counts()

tradingstrategy1returns_df = generate_returns(tradingstrategy1_df)

tradingstrategy1returns_df.to_csv('tradingstrategy1returns.csv', index=False)

# spy_df = yf.download('SPY', start=tradingstrategy1returns_df['Buy Date'].iloc[0], end=tradingstrategy1returns_df['Sell Date'].iloc[-1])
# spy_df['Cumulative Returns %'] = (spy_df['Close'] / spy_df['Close'].iloc[0]) * 100
# spy_df['Cumulative Returns %'].plot(figsize=(20, 10), title='SPY Cumulative Returns %')

# Plot the cumulative returns of the trading strategy against the SPY ETF
plt.figure(figsize=(20, 10))
plt.plot(tradingstrategy1returns_df['Sell Date'], tradingstrategy1returns_df['Cumulative Returns %'], label='Trading Strategy Cumulative Returns %', color='blue')
# plt.plot(spy_df['Cumulative Returns %'], label='SPY Cumulative Returns %', color='black')
# plt.title('Trading Strategy vs SPY Cumulative Returns %')
plt.legend()
plt.show()
plt.close()


# %%

def generate_performance(dataframe: pd.DataFrame):
    performance_df = dataframe.copy(deep=True)
    performance_df['Winning Trades'] = performance_df['Profitable'].apply(lambda x: 1 if x == 'Yes' else 0)
    performance_df['Losing Trades'] = performance_df['Profitable'].apply(lambda x: 1 if x == 'No' else 0)
    performance_df['Winning Trades'] = performance_df['Winning Trades'].cumsum()
    performance_df['Losing Trades'] = performance_df['Losing Trades'].cumsum()
    performance_df['Trade Number'] = range(1, len(performance_df) + 1)
    performance_df['Total Trades'] = performance_df['Trade Number']
    performance_df['Win Rate'] = (performance_df['Winning Trades'] / performance_df['Total Trades']) * 100
    performance_df['Loss Rate'] = (performance_df['Losing Trades'] / performance_df['Total Trades']) * 100
    performance_df['Win Rate'] = performance_df['Win Rate'].round(2)
    performance_df['Loss Rate'] = performance_df['Loss Rate'].round(2)
    performance_df['Win Rate'] = performance_df['Win Rate'].shift(1)
    performance_df['Loss Rate'] = performance_df['Loss Rate'].shift(1)
    performance_df['Win Rate'].iloc[0] = 0
    performance_df['Loss Rate'].iloc[0] = 0
    performance_df['Win Rate'] = performance_df['Win Rate'].fillna(0)
    performance_df['Loss Rate'] = performance_df['Loss Rate'].fillna(0)
    performance_df['Win Rate'] = performance_df['Win Rate'].astype(str) + '%'
    performance_df['Loss Rate'] = performance_df['Loss Rate'].astype(str) + '%'
    return performance_df

# %%
tradingstrategy1performance_df = generate_performance(tradingstrategy1returns_df)

tradingstrategy1performance_df.to_csv('tradingstrategy1performance_df.csv', index=False)

# %% [markdown]
# ---
# # Positions
# ---
# 
# ## Define Positions class

# %%
class Positions:
    def __init__(self, trade_choices: pd.DataFrame):
        self.positions = {}
        self.cash = 10000
        self.equity = 0
        self.profit = 0
        self.total_value = 0
        self.total_profit = 0
        self.total_equity = 0
        self.trades = {ticker, buy_date, buy_price, shares, sell_date, sell_price, profit}

    def buy(self, ticker: Ticker, amount: int):
        if self.cash < amount:
            print("Not enough cash to buy ", ticker.symbol)
        else:
            self.cash -= amount
            self.equity += amount
            self.positions[ticker.symbol] = amount
            print("Bought ", amount, " shares of ", ticker.symbol)

    def sell(self, ticker: Ticker, amount: int):
        if ticker.symbol not in self.positions:
            print("No position in ", ticker.symbol)
        elif self.positions[ticker.symbol] < amount:
            print("Not enough shares to sell")
        else:
            self.cash += amount
            self.equity -= amount
            self.positions[ticker.symbol] -= amount
            print("Sold ", amount, " shares of ", ticker.symbol)

    # def calculate_profit(self, ticker: Ticker):
    #     self.profit = (ticker.df['Adj Close'].iloc[-1] - ticker.df['Adj Close'].iloc[0]) * self.positions[ticker.symbol]
    #     return self.profit

    # def calculate_total_value(self):
    #     self.total_value = self.cash + self.equity
    #     return self.total_value

    # def calculate_total_profit(self):
    #     self.total_profit = self.total_value - 1000000
    #     return self.total_profit
    
    #     def buy_and_hold(self):
    #     self.df['Buy and Hold'] = self.df['Adj Close'] / self.df['Adj Close'].iloc[0]
    #     return self.df

    # def generate_returns(self):
    #     self.df['Returns'] = self.df['Adj Close'].pct_change()
    #     self.df.dropna(inplace=True)
    #     return self.df

    # def generate_sharpe_ratio(self):
    #     self.sharpe_ratio = (self.df['Returns'].mean() - 0.015) / self.df['Returns'].std()
    #     return self.sharpe_ratio

# %%


# aapl_ta.generate_prophet_forecast(frame = 'Adj Close', period=90)
# aapl_ta.generate_prophet_forecast(frame = 'Open')

# aapl_ta.autocorrelation_plot()

# aapl_ta.generate_correlation_matrix()

# %% [markdown]
# 

# %% [markdown]
# # [FB Prophet](https://facebook.github.io/prophet/)
# "Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well."



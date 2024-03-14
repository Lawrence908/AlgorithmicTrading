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
import plotly
from tickerTA import Ticker
from tickerTA import TechnicalAnalysis
from tradingStrategy1 import TradingStrategy1

# This is to avoid the SettingWithCopyWarning in pandas
pd.options.mode.copy_on_write = True 

# class Positions:
#     def __init__(self, trade_choices: pd.DataFrame):
#         self.positions = {}
#         self.cash = 10000
#         self.equity = 0
#         self.profit = 0
#         self.total_value = 0
#         self.total_profit = 0
#         self.total_equity = 0
#         self.trades = {ticker, buy_date, buy_price, shares, sell_date, sell_price, profit}

#     def buy(self, ticker: Ticker, amount: int):
#         if self.cash < amount:
#             print("Not enough cash to buy ", ticker.symbol)
#         else:
#             self.cash -= amount
#             self.equity += amount
#             self.positions[ticker.symbol] = amount
#             print("Bought ", amount, " shares of ", ticker.symbol)

#     def sell(self, ticker: Ticker, amount: int):
#         if ticker.symbol not in self.positions:
#             print("No position in ", ticker.symbol)
#         elif self.positions[ticker.symbol] < amount:
#             print("Not enough shares to sell")
#         else:
#             self.cash += amount
#             self.equity -= amount
#             self.positions[ticker.symbol] -= amount
#             print("Sold ", amount, " shares of ", ticker.symbol)

#     # def calculate_profit(self, ticker: Ticker):
#     #     self.profit = (ticker.df['Adj Close'].iloc[-1] - ticker.df['Adj Close'].iloc[0]) * self.positions[ticker.symbol]
#     #     return self.profit

#     # def calculate_total_value(self):
#     #     self.total_value = self.cash + self.equity
#     #     return self.total_value

#     # def calculate_total_profit(self):
#     #     self.total_profit = self.total_value - 1000000
#     #     return self.total_profit
    
#     #     def buy_and_hold(self):
#     #     self.df['Buy and Hold'] = self.df['Adj Close'] / self.df['Adj Close'].iloc[0]
#     #     return self.df

#     # def generate_returns(self):
#     #     self.df['Returns'] = self.df['Adj Close'].pct_change()
#     #     self.df.dropna(inplace=True)
#     #     return self.df

#     # def generate_sharpe_ratio(self):
#     #     self.sharpe_ratio = (self.df['Returns'].mean() - 0.015) / self.df['Returns'].std()
#     #     return self.sharpe_ratio

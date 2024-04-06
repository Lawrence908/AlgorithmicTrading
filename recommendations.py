import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import pytictoc as tt
from tickers500 import tickers500
from tickerTA import Ticker
from tickerTA import TechnicalAnalysis
from tradingStrategy1 import TradingStrategy1

# This is to avoid the SettingWithCopyWarning in pandas
pd.options.mode.copy_on_write = True 

# This class is used to parse the daily data from the Yahoo Finance API
# to implement the Ticker class, TechnicalAnalysis class and TradingStrategy1 class
# After processing all the data, it will output the current Buy/Sell recommendations list
# for the 500 S&P companies
# Also it will include the price change since the last recommendation

class Recommendations:
    def __init__(self, start = pd.to_datetime('today') - pd.DateOffset(months=6), end = pd.to_datetime('today')):
        self.start = start
        self.end = end
        self.tickers500 = tickers500
        self.buy_recommendations_df = pd.DataFrame()
        self.sell_recommendations_df = pd.DataFrame()
        self.get_stock_data()
        self.current_recommendations()

    def get_stock_data(self):
        self.tradingstrategy1_df = pd.DataFrame()
        self.signals_df = pd.DataFrame()
        for ticker in self.tickers500:
            stock = Ticker(ticker, self.start, self.end)
            # stock_df = stock_df._append(stock.df)
            techA = TechnicalAnalysis(stock)
            tradingS = TradingStrategy1(techA)
            self.signals_df = self.signals_df._append(tradingS.all_signals_df)
            self.tradingstrategy1_df = self.tradingstrategy1_df._append(tradingS.trades_df)
        self.signals_df = self.signals_df.sort_values(by='Date')

    def current_recommendations(self):
        for _, row in self.signals_df.iterrows():
            if row['Signal'] == 'Buy':
                self.buy_recommendations_df = self.buy_recommendations_df.append(row)
            elif row['Signal'] == 'Sell':
                self.sell_recommendations_df = self.sell_recommendations_df.append(row)
        self.buy_recommendations_df = self.buy_recommendations_df.sort_values(by='Date')
        self.sell_recommendations_df = self.sell_recommendations_df.sort_values(by='Date')
        # self.buy_recommendations_df = self.buy_recommendations_df.drop_duplicates(subset=['Ticker'])
        # self.sell_recommendations_df = self.sell_recommendations_df.drop_duplicates(subset=['Ticker'])
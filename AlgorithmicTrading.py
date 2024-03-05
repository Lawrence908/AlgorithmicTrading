import yfinance as yf

import finplot as fplt #run 'sudo apt-get install -y libgl1-mesa-glx' in terminal to install if issues arise

import yahoo_fin.stock_info as yfin
from yfinance import Tickers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import pandas_datareader as pdr
import pytictoc as tt

class Ticker:
    def __init__(self, symbol, start=pd.to_datetime('today') - pd.DateOffset(months=12), end=pd.to_datetime('today')):
        self.symbol = symbol
        self.start = start
        self.end = end
        if '.' in self.symbol:
            self.symbol = self.symbol.replace('.', '-')
        self.df = yf.download(self.symbol, self.start, self.end, progress=False)
        if self.df.empty:
            print("No data found for ", self.symbol)
            sys.exit(1)
        else:
            self.Ticker = yf.Ticker(self.symbol)
            self.actions = self.Ticker.get_actions()
            # self.analysis = self.Ticker.get_analysis()
            self.balance = self.Ticker.get_balance_sheet()
            self.calendar = self.Ticker.get_calendar()
            self.cf = self.Ticker.get_cashflow()
            self.info = self.Ticker.get_info()
            self.inst_holders = self.Ticker.get_institutional_holders()
            self.news = self.Ticker.get_news()
            self.recommendations = self.Ticker.get_recommendations()
            # self.sustainability = self.Ticker.get_sustainability()

    def __str__(self):
        return self.symbol
    
    def __repr__(self):
        return self.symbol

if __name__ == "__main__":
    current = Ticker('NVDA')
    current.info
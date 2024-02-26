import sys
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ta
import pandas_datareader as pdr

class Backtest:

    def __init__(self, symbol):
        # self.ticker_list = []
        self.symbol = symbol
        self.start = '2022-01-31'
        self.end = '2024-01-31'
        if '.' in self.symbol:
            self.symbol = self.symbol.replace('.','-')
        self.df = yf.download(self.symbol, self.start, self.end, progress=False)
        if self.df.empty:
            print("No data found for ", symbol)
            sys.exit(1)
        else:
            self.calc_indicators()
            self.generate_signals()
            self.filter_signals_non_overlapping()
            self.profit = self.calc_profit()
            self.max_drawdown = self.profit.min()
            self.cumulative_profit = (self.profit + 1).prod() - 1
            self.plot_chart()


    def calc_indicators(self):
        self.df['ma_20'] = self.df['Adj Close'].rolling(window=20).mean()
        self.df['vol'] = self.df['Adj Close'].rolling(window=20).std()
        self.df['upper_bb'] = self.df['ma_20'] + (self.df['vol'] * 2)
        self.df['lower_bb'] = self.df['ma_20'] - (self.df['vol'] * 2)
        self.df['rsi'] = ta.momentum.rsi(self.df['Adj Close'], window=6)
        self.df.dropna(inplace=True)

    def generate_signals(self):
        conditions = [(self.df.rsi < 30) & (self.df['Adj Close'] < self.df.lower_bb),
                        (self.df.rsi > 70) & (self.df['Adj Close'] > self.df.upper_bb)]
        choices = ['Buy', 'Sell']
        self.df['signal'] = np.select(conditions, choices)
        self.df.signal = self.df['signal'].shift(1)
        self.df.dropna(inplace=True)

    def filter_signals_non_overlapping(self):
        position = False
        buy_dates, sell_dates = [], []

        for index, row in self.df.iterrows():
            if not position and row.signal == 'Buy':
                position = True
                buy_dates.append(index)

            if position and row.signal == 'Sell':
                position = False
                sell_dates.append(index)

        self.buy_arr = self.df.loc[buy_dates].Open
        self.sell_arr = self.df.loc[sell_dates].Open

    def calc_profit(self):
        if self.buy_arr.index[-1] > self.sell_arr.index[-1]:
            self.buy_arr = self.buy_arr[:-1]
        return (self.sell_arr.values - self.buy_arr.values)/self.buy_arr.values

    def plot_chart(self):
        plt.figure(figsize=(20, 10))
        plt.plot(self.df['Adj Close'], label=self.symbol, alpha=1)
        plt.plot(self.df['ma_20'], label='20 Day Moving Average', alpha=0.8)
        plt.plot(self.df['upper_bb'], label='Upper Bollinger Band', alpha=0.55)
        plt.plot(self.df['lower_bb'], label='Lower Bollinger Band', alpha=0.50)
    
        # plt.plot(self.cumulative_profit * 100, label='Cumulative Profit', alpha=0.50)
        # plt.xlim(pd.to_datetime(self.start), pd.to_datetime(self.end))

        plt.scatter(self.buy_arr.index, self.buy_arr.values, label='Buy Signal', marker='^', color='green')
        plt.scatter(self.sell_arr.index, self.sell_arr.values, label='Sell Signal', marker='v', color='red')
        plt.title(self.symbol + ' Backtest')
        plt.legend(loc='upper left')
        plt.savefig('figures/backTest/' + self.symbol + ' backtest.png')
        plt.show()

instance = Backtest('AAPL')

print(instance.profit)

print(instance.max_drawdown)

print(instance.cumulative_profit)
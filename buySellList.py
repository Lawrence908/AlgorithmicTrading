import sys
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ta
import pandas_datareader as pdr
import pytictoc as tt

class Signals:

    def __init__(self, symbol, start = pd.to_datetime('today') - pd.DateOffset(months=12), end = pd.to_datetime('today')):
        self.symbol = symbol
        self.start = start
        self.end = end
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
            self.cumulative_profit = (self.profit + 1).prod() - 1
            self.max_drawdown = self.cumulative_profit.min()
            self.plot_chart()
            # self.trades_df()
            # self.plot_trades()
            # self.calc_buy_and_hold()
            self.recent_trade_df()


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
        self.df['signal_strength'] = np.where(self.df['signal'] == 'Buy', self.df['Adj Close'] - self.df['lower_bb'], self.df['upper_bb'] - self.df['Adj Close'])


    def filter_signals_non_overlapping(self):
        position = False
        buy_dates, sell_dates = [], []

        for index, row in self.df.iterrows():
            if not position and row.signal == 'Buy':
                position = True
                buy_dates.append(index)

# Add in a stop loss condition
            if position and row.signal == 'Sell':
                position = False
                sell_dates.append(index)

        self.buy_arr = self.df.loc[buy_dates].Open
        self.sell_arr = self.df.loc[sell_dates].Open


    def calc_profit(self):
        if len(self.buy_arr) == 0 or len(self.sell_arr) == 0:
            return np.array([])  # Return an empty array if buy_arr or sell_arr is empty
        if self.buy_arr.index[-1] > self.sell_arr.index[-1]:
            self.buy_arr = self.buy_arr[:-1]
        return (self.sell_arr.values - self.buy_arr.values)/self.buy_arr.values


    def plot_chart(self):
        plt.figure(figsize=(20, 10))
        plt.plot(self.df['Adj Close'], label=self.symbol, alpha=1)
        plt.plot(self.df['ma_20'], label='20 Day Moving Average', alpha=0.8)
        plt.plot(self.df['upper_bb'], label='Upper Bollinger Band', alpha=0.55)
        plt.plot(self.df['lower_bb'], label='Lower Bollinger Band', alpha=0.50)
        plt.scatter(self.buy_arr.index, self.buy_arr.values, label='Buy Signal', marker='^', color='green')
        plt.scatter(self.sell_arr.index, self.sell_arr.values, label='Sell Signal', marker='v', color='red')
        for i in range(len(self.buy_arr)):
            plt.text(self.buy_arr.index[i], self.df.loc[self.buy_arr.index[i]]['Adj Close'], "      $" + str(round(self.buy_arr.values[i], 2)), fontsize=10, color='g')
        for i in range(len(self.sell_arr)):
            plt.text(self.sell_arr.index[i], self.df.loc[self.sell_arr.index[i]]['Adj Close'], "      $" + str(round(self.sell_arr.values[i], 2)), fontsize=10, color='r')
        plt.title(self.symbol + ' Recent Trades ' + str(self.start.date()) + ' to ' + str(self.end.date()))
        plt.legend(loc='upper left')
        plt.savefig('figures/trades/' + self.symbol + ' ' + str(self.end.date()) + '.png')
        # plt.show()
        plt.close()


    def recent_trade_df(self):
        self.recent_trade_df = pd.DataFrame({})
        if len(self.buy_arr) > 0:
            self.recent_trade_df = self.recent_trade_df._append({'Ticker': self.symbol, 'Trade': 'BUY', 'Date': self.buy_arr.index[-1], 'Price': self.buy_arr.values[-1],}, ignore_index=True)
            self.recent_trade_df['Signal Strength'] = self.df.loc[self.recent_trade_df['Date']]['signal_strength'].values
        if len(self.sell_arr) > 0:
            self.recent_trade_df = self.recent_trade_df._append({'Ticker': self.symbol, 'Trade': 'SELL', 'Date': self.sell_arr.index[-1], 'Price': self.sell_arr.values[-1]}, ignore_index=True)
            self.recent_trade_df['Signal Strength'] = self.df.loc[self.recent_trade_df['Date']]['signal_strength'].values


    def trades_df(self):
        self.trades_df = pd.DataFrame({'Ticker': self.symbol, 'Buy Date': self.buy_arr.index, 'Buy Price': self.buy_arr.values, 'Sell Date': self.sell_arr.index, 'Sell Price': self.sell_arr.values})
        # self.trades_df.insert(0, 'Ticker', self.symbol)
        self.trades_df['Profit'] = self.trades_df['Sell Price'] - self.trades_df['Buy Price']
        self.trades_df['Profit %'] = (self.trades_df['Profit'] / self.trades_df['Buy Price']) * 100
        self.trades_df['Duration'] = self.trades_df['Sell Date'] - self.trades_df['Buy Date']
        self.trades_df['Duration'] = self.trades_df['Duration'].dt.days
        self.trades_df['Ticker Cum Profit'] = self.trades_df['Profit'].cumsum()
        if len(self.trades_df['Buy Price']) > 0:
            self.trades_df['Ticker Cum Profit %'] = (self.trades_df['Ticker Cum Profit'] / self.trades_df['Buy Price'].iloc[0]) * 100
        else:
            self.trades_df['Ticker Cum Profit %'] = 0
        self.trades_df['Profitable'] = self.trades_df['Profit'] > 0
        self.trades_df['Profitable'] = self.trades_df['Profitable'].replace({True: 'Yes', False: 'No'})
        self.trades_df['Trade Number'] = range(1, len(self.trades_df) + 1)
        self.trades_df['RSI'] = self.df.loc[self.trades_df['Buy Date']]['rsi'].values
        self.trades_df['Upper BB'] = self.df.loc[self.trades_df['Buy Date']]['upper_bb'].values
        self.trades_df['Lower BB'] = self.df.loc[self.trades_df['Buy Date']]['lower_bb'].values
        self.trades_df['Vol'] = self.df.loc[self.trades_df['Buy Date']]['vol'].values

    def calc_buy_and_hold(self):
        self.df['Control'] = self.df['Close'] - self.df['Open']
        self.df['Control %'] = (self.df['Control'] / self.df['Open']) * 100
        self.df['Control Cumulative %'] = self.df['Control %'].cumsum()


    def plot_trades(self):
        plt.figure(figsize=(20, 10))
        plt.plot(self.trades_df['Buy Date'], self.trades_df['Ticker Cum Profit %'], drawstyle="steps-post", label=self.symbol, alpha=1)
        for i in range(len(self.trades_df)):
            plt.text(self.trades_df['Buy Date'].iloc[i], self.trades_df['Ticker Cum Profit %'].iloc[i], "      " + str(round(self.trades_df['Profit %'].iloc[i], 2)) + "%", fontsize=10, color='black')
        spy_df = yf.download('SPY', self.start, self.end, progress=False)
        spy_df['Control'] = spy_df['Close'] - spy_df['Open']
        spy_df['Control %'] = (spy_df['Control'] / spy_df['Open']) * 100
        spy_df['Control Cumulative %'] = spy_df['Control %'].cumsum()
        plt.plot(spy_df['Control Cumulative %'], drawstyle="steps-post", label='SPY', alpha=0.8)

        plt.title(self.symbol + ' Ticker Cum Profit %')
        plt.legend(loc='upper left')
        # plt.savefig('figures/profit/' + self.symbol + ' ' + str(self.end.date()) + '.png')
        # plt.show()
        plt.close()


if __name__ == "__main__":
    t = tt.TicToc()
    t.tic() # start timer

    # This is to avoid the SettingWithCopyWarning in pandas
    pd.options.mode.copy_on_write = True 

    # Read the filename provided in command line arg,
    # store the tickers in a list
    with open(sys.argv[1], 'r') as f:
        ticker_list = f.read().splitlines()

    # Create a dataframe to store all trades for all tickers
    trades_df = pd.DataFrame()

    # Create a dataframe to store the most recent trade for all tickers
    recent_trade_df = pd.DataFrame()

    # Loop through the tickers and create a Signals object for each ticker
    for count, ticker in enumerate(ticker_list):
        SIG = Signals(ticker)
        # trades_df = trades_df._append(SIG.trades_df)
        try:
            recent_trade_df = recent_trade_df._append(SIG.recent_trade_df.iloc[-1])
        except:
            pass

    # Sort recent_trades_df by Date descending and reset the index
    recent_trade_df.sort_values(by='Date', ascending=False, inplace=True)
    recent_trade_df.reset_index(drop=True, inplace=True)


    # print(trades_df)
    print(recent_trade_df)

    #Save recent_trade_df to a csv file
    recent_trade_df.to_csv('CSV/recent_trades.csv', index=False)


    t.toc() #elapsed time
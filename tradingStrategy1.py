import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from tickerTA import TechnicalAnalysis

# This is to avoid the SettingWithCopyWarning in pandas
pd.options.mode.copy_on_write = True 

class TradingStrategy1:
    def __init__(self, ta: TechnicalAnalysis):
        self.ta = ta
        self.start = self.ta.start
        self.end = self.ta.end
        self.Ticker = self.ta.Ticker
        self.symbol = self.ta.symbol
        self.df = self.ta.df_ta.copy(deep=True)
        self.all_signals_df = pd.DataFrame(columns=['Ticker', 'Buy/Sell', 'Date', 'Price'])
        self.generate_signal()
        self.generate_trades()
        self.calc_stats()
        self.plot_trades()
        self.plot_profit()


    def __str__(self):
        return self.symbol
    
    def __repr__(self):
        return self.symbol
    
    def generate_signal(self):
        conditions = [(self.df['rsi'] < 30) & (self.df['Adj Close'] < self.df['BollingerBandLow']),
                        (self.df['rsi'] > 70) & (self.df['Adj Close'] > self.df['BollingerBandHigh'])] 
        choices = ['Buy', 'Sell']
        self.df['Signal'] = np.select(conditions, choices, default='Hold')
        self.df.dropna(inplace=True)
        self.df['Signal'] = self.df['Signal'].shift(1)

    def generate_trades(self):
        self.position = False
        self.buy_dates, self.sell_dates = [], []
        self.buy_prices, self.sell_prices = [], []

        for index, row in self.df.iterrows():
            if row['Signal'] == 'Buy' and self.position == False:
                self.buy_dates.append(index)
                self.buy_prices.append(row['Open'])
                self.position = True
                new_row1 = pd.Series({'Ticker': self.symbol, 'Buy/Sell': 'Buy', 'Date': index, 'Price': row['Open']})
                new_row1_df = new_row1.to_frame().T
                self.all_signals_df = self.all_signals_df._append(new_row1_df, ignore_index=True)
            elif row['Signal'] == 'Sell' and self.position == True:
                self.sell_dates.append(index)
                self.sell_prices.append(row['Open'])
                self.position = False
                new_row2 = pd.Series({'Ticker': self.symbol, 'Buy/Sell': 'Sell', 'Date': index, 'Price': row['Open']})
                new_row2_df = new_row2.to_frame().T
                self.all_signals_df = self.all_signals_df._append(new_row2_df, ignore_index=True)
                
        self.buy_arr = self.df.loc[self.buy_dates].Open
        self.sell_arr = self.df.loc[self.sell_dates].Open

        if len(self.buy_arr) != len(self.sell_arr):
            if len(self.buy_arr) > len(self.sell_arr):
                self.buy_arr = self.buy_arr[:-1]
            else:
                self.sell_arr = self.sell_arr[:-1]

        self.trades_df = pd.DataFrame({'Buy Date': self.buy_arr.index, 'Buy Price': self.buy_arr.values})
        self.trades_df.insert(0, 'Ticker', self.symbol)
        try:
            self.trades_df['Buy P/E Ratio'] = self.df.loc[self.trades_df['Buy Date']]['P/E Ratio'].values
        except:
            self.trades_df['Buy P/E Ratio'] = 0
        try:
            self.trades_df['Buy Fwd P/E Ratio'] = self.df.loc[self.trades_df['Buy Date']]['Fwd P/E Ratio'].values
        except:
            self.trades_df['Buy Fwd P/E Ratio'] = 0
        try:
            self.trades_df['Buy P/B Ratio'] = self.df.loc[self.trades_df['Buy Date']]['P/B Ratio'].values
        except:
            self.trades_df['Buy P/B Ratio'] = 0
        self.trades_df['Buy RSI'] = self.df.loc[self.trades_df['Buy Date']]['rsi'].values
        self.trades_df['Buy Upper BB'] = self.df.loc[self.trades_df['Buy Date']]['upper_bb'].values
        self.trades_df['Buy Bollinger %b'] = self.df.loc[self.trades_df['Buy Date']]['Bollinger%b'].values
        self.trades_df['Buy Lower BB'] = self.df.loc[self.trades_df['Buy Date']]['lower_bb'].values
        self.trades_df['Buy vol'] = self.df.loc[self.trades_df['Buy Date']]['vol'].values
        self.trades_df['Buy Z Score'] = self.df.loc[self.trades_df['Buy Date']]['Z Score Adj Close'].values
        self.trades_df['Buy MACD'] = self.df.loc[self.trades_df['Buy Date']]['MACD'].values
        self.trades_df['Buy VWAP'] = self.df.loc[self.trades_df['Buy Date']]['VWAP'].values
        self.trades_df['Buy OBV'] = self.df.loc[self.trades_df['Buy Date']]['OBV'].values
        self.trades_df['Buy Stoch'] = self.df.loc[self.trades_df['Buy Date']]['Stoch'].values
        self.trades_df['Buy Awesome Oscillator'] = self.df.loc[self.trades_df['Buy Date']]['Awesome Oscillator'].values
        self.trades_df['Buy Ultimate Oscillator'] = self.df.loc[self.trades_df['Buy Date']]['Ultimate Oscillator'].values
        self.trades_df['Buy TSI'] = self.df.loc[self.trades_df['Buy Date']]['TSI'].values
        self.trades_df['Buy Acum/Dist'] = self.df.loc[self.trades_df['Buy Date']]['Accumulation/Distribution'].values
        self.trades_df['Sell Date'] = self.sell_arr.index
        self.trades_df['Sell Price'] = self.sell_arr.values
        try:
            self.trades_df['Sell P/E Ratio'] = self.df.loc[self.trades_df['Sell Date']]['P/E Ratio'].values
        except:
            self.trades_df['Sell P/E Ratio'] = 0
        try:
            self.trades_df['Sell Fwd P/E Ratio'] = self.df.loc[self.trades_df['Sell Date']]['Fwd P/E Ratio'].values
        except:
            self.trades_df['Sell Fwd P/E Ratio'] = 0
        try:
            self.trades_df['Sell P/B Ratio'] = self.df.loc[self.trades_df['Sell Date']]['P/B Ratio'].values
        except:
            self.trades_df['Sell P/B Ratio'] = 0
        self.trades_df['Sell RSI'] = self.df.loc[self.trades_df['Sell Date']]['rsi'].values
        self.trades_df['Sell Upper BB'] = self.df.loc[self.trades_df['Sell Date']]['upper_bb'].values
        self.trades_df['Sell Bollinger %b'] = self.df.loc[self.trades_df['Sell Date']]['Bollinger%b'].values
        self.trades_df['Sell Lower BB'] = self.df.loc[self.trades_df['Sell Date']]['lower_bb'].values
        self.trades_df['Sell vol'] = self.df.loc[self.trades_df['Sell Date']]['vol'].values
        self.trades_df['Sell Z Score'] = self.df.loc[self.trades_df['Sell Date']]['Z Score Adj Close'].values
        self.trades_df['Sell MACD'] = self.df.loc[self.trades_df['Sell Date']]['MACD'].values
        self.trades_df['Sell VWAP'] = self.df.loc[self.trades_df['Sell Date']]['VWAP'].values
        self.trades_df['Sell OBV'] = self.df.loc[self.trades_df['Sell Date']]['OBV'].values
        self.trades_df['Sell Stoch'] = self.df.loc[self.trades_df['Sell Date']]['Stoch'].values
        self.trades_df['Sell Awesome Oscillator'] = self.df.loc[self.trades_df['Sell Date']]['Awesome Oscillator'].values
        self.trades_df['Sell Ultimate Oscillator'] = self.df.loc[self.trades_df['Sell Date']]['Ultimate Oscillator'].values
        self.trades_df['Sell TSI'] = self.df.loc[self.trades_df['Sell Date']]['TSI'].values
        self.trades_df['Sell Acum/Dist'] = self.df.loc[self.trades_df['Sell Date']]['Accumulation/Distribution'].values
        self.trades_df['Profit'] = self.trades_df['Sell Price'] - self.trades_df['Buy Price']
        self.trades_df['Profit %'] = (self.trades_df['Profit'] / self.trades_df['Buy Price']) * 100
        self.trades_df['Profit %'] = self.trades_df['Profit %'].round(2)
        self.trades_df['Duration'] = self.trades_df['Sell Date'] - self.trades_df['Buy Date']
        self.trades_df['Duration'] = self.trades_df['Duration'].dt.days
        self.trades_df['Ticker Cum Profit'] = self.trades_df['Profit'].cumsum()
        if len(self.trades_df['Buy Price']) > 0:
            self.trades_df['Ticker Cum Profit %'] = (self.trades_df['Ticker Cum Profit'] / self.trades_df['Buy Price'].iloc[0]) * 100
        else:
            self.trades_df['Ticker Cum Profit %'] = 0
        self.trades_df['Ticker Cum Profit %'] = self.trades_df['Ticker Cum Profit %'].round(2)
        self.trades_df['Profitable'] = self.trades_df['Profit'] > 0
        self.trades_df['Profitable'] = self.trades_df['Profitable'].replace({True: 'Yes', False: 'No'})
        return self.trades_df

    def plot_trades(self):
        plt.figure(figsize=(20, 10))
        plt.plot(self.df['Adj Close'], label=self.symbol, alpha=1)
        plt.plot(self.df['BollingerMAvg'], label='Bollinger Moving Avg(20)', alpha=0.45)
        plt.plot(self.df['BollingerBandHigh'], label='Bollinger Band High', alpha=0.55)
        plt.plot(self.df['BollingerBandLow'], label='Bollinger Band Low', alpha=0.55)
        plt.scatter(self.buy_arr.index, self.buy_arr.values, label='Buy Signal', marker='^', color='green')
        plt.scatter(self.sell_arr.index, self.sell_arr.values, label='Sell Signal', marker='v', color='red')
        for i in range(len(self.buy_arr)):
            plt.text(self.buy_arr.index[i], self.buy_arr.values[i], "      $" + str(round(self.buy_arr.values[i], 2)), fontsize=10, color='g')
        for i in range(len(self.sell_arr)):
            plt.text(self.sell_arr.index[i], self.sell_arr.values[i], "      $" + str(round(self.sell_arr.values[i], 2)), fontsize=10, color='r')
        plt.title(self.symbol + ' Trading Strategy')
        plt.legend()
        # plt.savefig('tradingStrategy1/trades/' + self.symbol + '.png')
        plt.show()
        plt.close()

    def plot_profit(self):
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
        # plt.savefig('tradingStrategy1/profit/' + self.symbol + '.png')
        plt.show()
        plt.close()

    def calc_stats(self):
        self.winning_trades = self.trades_df['Profitable'].value_counts().get('Yes', 0)
        self.losing_trades = self.trades_df['Profitable'].value_counts().get('No', 0)
        self.total_trades = self.winning_trades + self.losing_trades
        self.win_rate = self.trades_df['Profitable'].value_counts(normalize=True).get('Yes', 0) * 100
        self.loss_rate = self.trades_df['Profitable'].value_counts(normalize=True).get('No', 0) * 100
        self.win_rate = round(self.win_rate, 2)
        self.loss_rate = round(self.loss_rate, 2)
        self.sharpe_ratio = self.trades_df['Profit %'].mean() / self.trades_df['Profit %'].std()
        self.sharpe_ratio = round(self.sharpe_ratio, 2)
        self.avg_profit = self.trades_df['Profit %'].mean()
        self.avg_profit = round(self.avg_profit, 2)
        self.avg_duration = self.trades_df['Duration'].mean()
        self.avg_duration = round(self.avg_duration, 2)
        self.total_profit = self.trades_df['Profit'].sum()
        self.total_profit = round(self.total_profit, 2)
        self.total_profit_percent = self.trades_df['Profit %'].sum()
        self.total_profit_percent = round(self.total_profit_percent, 2)

        self.stats = {
            'Total Trades': self.total_trades,
            'Winning Trades': self.winning_trades,
            'Losing Trades': self.losing_trades,
            'Win Rate %': self.win_rate,
            'Loss Rate %': self.loss_rate,
            'Sharpe Ratio': self.sharpe_ratio,
            'Average Profit %': self.avg_profit,
            'Average Duration': self.avg_duration,
            'Total Profit': self.total_profit,
            'Total Profit %': self.total_profit_percent
        }
        
        return self.stats
    
    def get_buytable(self):
        self.buytable = self.trades_df[['Buy P/E Ratio', 'Buy Fwd P/E Ratio', 'Buy P/B Ratio', 'Buy RSI', 'Buy Upper BB', 'Buy Bollinger %b', 'Buy Lower BB', 'Buy vol', 'Buy Z Score', 'Buy MACD', 'Buy VWAP', 'Buy OBV', 'Buy Stoch', 'Buy Awesome Oscillator', 'Buy Ultimate Oscillator', 'Buy TSI', 'Buy Acum/Dist', 'Profitable']]
        return self.buytable

    def get_selltable(self):
        self.selltable = self.trades_df[['Sell P/E Ratio', 'Sell Fwd P/E Ratio', 'Sell P/B Ratio', 'Sell RSI', 'Sell Upper BB', 'Sell Bollinger %b', 'Sell Lower BB', 'Sell vol', 'Sell Z Score', 'Sell MACD', 'Sell VWAP', 'Sell OBV', 'Sell Stoch', 'Sell Awesome Oscillator', 'Sell Ultimate Oscillator', 'Sell TSI', 'Sell Acum/Dist', 'Profitable']]
        return self.selltable
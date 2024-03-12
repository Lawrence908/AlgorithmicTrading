import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from tickerTA import TechnicalAnalysis

class TradingStrategy1:
    def __init__(self, ta: TechnicalAnalysis):
        self.ta = ta
        self.Ticker = self.ta.Ticker
        self.symbol = self.ta.symbol
        self.df = self.ta.df_ta.copy(deep=True)
        self.generate_signal()
        self.generate_trades()
        # self.generate_returns()
        # self.generate_performance()

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

            elif row['Signal'] == 'Sell' and self.position == True:
                self.sell_dates.append(index)
                self.sell_prices.append(row['Open'])
                self.position = False

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

    def plot_chart(self):
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
        plt.show()
        plt.close()
    
    # def generate_returns(self):
    #     self.returns_df = self.trades_df.copy(deep=True)
    #     self.returns_df['Returns'] = self.returns_df['Sell Price'] - self.returns_df['Buy Price']
    #     self.returns_df['Returns %'] = (self.returns_df['Returns'] / self.returns_df['Buy Price']) * 100
    #     self.returns_df['Returns %'] = self.returns_df['Returns %'].round(2)
    #     self.returns_df['Returns %'] = self.returns_df['Returns %'].astype(str) + '%'
    #     self.returns_df['Cumulative Returns'] = self.returns_df['Returns'].cumsum()
    #     self.returns_df['Cumulative Returns %'] = (self.returns_df['Cumulative Returns'] / self.returns_df['Buy Price'].iloc[0]) * 100
    #     self.returns_df['Cumulative Returns %'] = self.returns_df['Cumulative Returns %'].shift(1)
    #     self.returns_df['Cumulative Returns %'].iloc[0] = 0
    #     self.returns_df['Cumulative Returns %'] = self.returns_df['Cumulative Returns %'].fillna(0)
    #     self.returns_df['Cumulative Returns %'] = self.returns_df['Cumulative Returns %'].round(2)
    #     self.returns_df['Cumulative Returns %'] = self.returns_df['Cumulative Returns %'].astype(str) + '%'
    #     self.returns_df['Trade Number'] = range(1, len(self.returns_df) + 1)
    #     self.returns_df['Profitable'] = self.returns_df['Returns'] > 0
    #     self.returns_df['Profitable'] = self.returns_df['Profitable'].replace({True: 'Yes', False: 'No'})
    #     return self.returns_df
    
    # def generate_performance(self):
    #     self.performance_df = self.returns_df.copy(deep=True)
    #     self.performance_df['Winning Trades'] = self.performance_df['Profitable'].apply(lambda x: 1 if x == 'Yes' else 0)
    #     self.performance_df['Losing Trades'] = self.performance_df['Profitable'].apply(lambda x: 1 if x == 'No' else 0)
    #     self.performance_df['Winning Trades'] = self.performance_df['Winning Trades'].cumsum()
    #     self.performance_df['Losing Trades'] = self.performance_df['Losing Trades'].cumsum()
    #     self.performance_df['Total Trades'] = self.performance_df['Trade Number']
    #     self.performance_df['Win Rate'] = (self.performance_df['Winning Trades'] / self.performance_df['Total Trades']) * 100
    #     self.performance_df['Loss Rate'] = (self.performance_df['Losing Trades'] / self.performance_df['Total Trades']) * 100
    #     self.performance_df['Win Rate'] = self.performance_df['Win Rate'].round(2)
    #     self.performance_df['Loss Rate'] = self.performance_df['Loss Rate'].round(2)
    #     self.performance_df['Win Rate'] = self.performance_df['Win Rate'].shift(1)
    #     self.performance_df['Loss Rate'] = self.performance_df['Loss Rate'].shift(1)
    #     self.performance_df['Win Rate'].iloc[0] = 0
    #     self.performance_df['Loss Rate'].iloc[0] = 0
    #     self.performance_df['Win Rate'] = self.performance_df['Win Rate'].fillna(0)
    #     self.performance_df['Loss Rate'] = self.performance_df['Loss Rate'].fillna(0)
    #     self.performance_df['Win Rate'] = self.performance_df['Win Rate'].astype(str) + '%'
    #     self.performance_df['Loss Rate'] = self.performance_df['Loss Rate'].astype(str) + '%'
    #     return self.performance_df


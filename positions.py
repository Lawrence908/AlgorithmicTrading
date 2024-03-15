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

class Position:
    def __init__(self, symbol, buy_date, buy_price, shares):
        self.symbol = symbol
        self.buy_date = buy_date
        self.buy_price = buy_price
        self.shares = shares

    def sell(self, sell_date, sell_price):
        self.sell_date = sell_date
        self.sell_price = sell_price

    def calculate_profit(self):
        return (self.sell_price - self.buy_price) * self.shares
    

class Positions:
    def __init__(self, starting_capital=100000, trading_fee=3.66):
        self.positions = {}
        self.trade_history = []
        self.trade_history_df = pd.DataFrame()
        self.starting_capital = starting_capital
        self.capital = starting_capital
        self.trading_fee = trading_fee
        self.total_profit = 0
        self.sharpe_ratio = 0
        self.win_rate = 0
        self.max_drawdown = 0
        self.sortino_ratio = 0

    def buy(self, symbol, buy_date, buy_price, shares):
        if symbol in self.positions:
            print(f'Position for {symbol} already exists')
            return
        if (buy_price * shares) + self.trading_fee > self.capital:
            print(f'Not enough capital to buy {shares} shares of {symbol}')
            return
        position = Position(symbol, buy_date, buy_price, shares)
        self.positions[symbol] = position
        self.capital -= (buy_price * shares) + self.trading_fee
        print("Bought", shares, "shares of", symbol, "at", buy_price, "on", buy_date)

    def sell(self, symbol, sell_date, sell_price):
        if symbol in self.positions:
            position = self.positions[symbol]
            position.sell(sell_date, sell_price)
            self.capital += (sell_price * position.shares) - self.trading_fee
            profit = position.calculate_profit()
            self.total_profit += profit
            profit_percent = (profit / (position.buy_price * position.shares)) * 100
            self.trade_history.append({
                'symbol': symbol,
                'buy_date': position.buy_date,
                'buy_price': position.buy_price,
                'sell_date': position.sell_date,
                'sell_price': position.sell_price,
                'shares': position.shares,
                'profit': profit,
                'profit_percent': profit_percent
            })
            self.trade_history_df = pd.DataFrame(self.trade_history)
            print("Sold", position.shares, "shares of", symbol, "at", sell_price, "on", sell_date)
            del self.positions[symbol]
        else:
            print(f'No position for {symbol}')

    def get_open_positions(self):
        open_positions = []
        for symbol, position in self.positions.items():
            open_positions.append({
                'symbol': symbol,
                'buy_date': position.buy_date,
                'buy_price': position.buy_price,
                'shares': position.shares
            })
        return open_positions

    def cancel_open_positions(self):
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            self.capital += (position.buy_price * position.shares) - self.trading_fee
            del self.positions[symbol]

    def get_daily_return(self):
        return (self.capital - self.starting_capital) / self.starting_capital
    
    def get_share_value(self, symbol):
        if symbol in self.positions:
            position = self.positions[symbol]
            return position.shares * position.buy_price
        else:
            return 0
    
    def get_avg_profit(self):
        return self.total_profit / len(self.trade_history)
    
    def get_avg_profit_percent(self):
        return (self.total_profit / self.starting_capital) * 100
    
    def plot_profit_frequency(self):
        profits = [trade['profit'] for trade in self.trade_history]
        plt.figure(figsize=(20, 10))
        plt.hist(profits, bins=20)
        plt.title('Profit Frequency')
        plt.xlabel('Profit')
        plt.ylabel('Frequency')
        plt.show()

    def plot_profit_percent_frequency(self):
        profits = [trade['profit_percent'] for trade in self.trade_history]
        plt.figure(figsize=(20, 10))
        plt.hist(profits, bins=20)
        plt.title('Profit Percent Frequency')
        plt.xlabel('Profit Percent')
        plt.ylabel('Frequency')
        plt.show()

    def plot_profit_distribution(self):
        profits = [trade['profit'] for trade in self.trade_history]
        plt.figure(figsize=(20, 10))
        plt.title('Profit Distribution')
        plt.xlabel('Profit')
        plt.ylabel('Density')
        sns.kdeplot(profits, fill=True)
 
        plt.show()

    def plot_profit_percent_distribution(self):
        profits = [trade['profit_percent'] for trade in self.trade_history]
        plt.figure(figsize=(20, 10))
        plt.title('Profit Percent Distribution')
        plt.xlabel('Profit Percent')
        plt.ylabel('Density')
        sns.kdeplot(profits, fill=True)

        plt.show()

    def plot_profit_time(self):
        profits = [trade['profit'] for trade in self.trade_history]
        dates = [trade['sell_date'] for trade in self.trade_history]
        plt.figure(figsize=(20, 10))
        plt.plot(dates, profits)
        plt.title('Profit Over Time')
        plt.xlabel('Date')
        plt.ylabel('Profit')
        plt.show()

    def plot_profit_percent_time(self):
        profits = [trade['profit_percent'] for trade in self.trade_history]
        dates = [trade['sell_date'] for trade in self.trade_history]
        plt.figure(figsize=(20, 10))
        plt.plot(dates, profits)
        plt.title('Profit Percent Over Time')
        plt.xlabel('Date')
        plt.ylabel('Profit Percent')
        plt.show()

    def plot_capital_time(self):
        capital = [self.starting_capital + trade['profit'] for trade in self.trade_history]
        dates = [trade['sell_date'] for trade in self.trade_history]
        plt.figure(figsize=(20, 10))
        plt.plot(dates, capital)
        plt.title('Capital Over Time')
        plt.xlabel('Date')
        plt.ylabel('Capital')
        plt.show()

    def plot_metrics(self):
        metrics = {
            'Sharpe Ratio': self.sharpe_ratio,
            'Win Rate': self.win_rate,
            'Max Drawdown': self.max_drawdown,
            'Sortino Ratio': self.sortino_ratio
        }
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Trading Metrics')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.show()
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        all_returns = []
        for trade in self.trade_history:
            buy_date = trade['buy_date']
            sell_date = trade['sell_date']
            buy_price = trade['buy_price']
            sell_price = trade['sell_price']
            shares = trade['shares']
            symbol = trade['symbol']
            position = Position(symbol, buy_date, buy_price, shares)
            position.sell(sell_date, sell_price)
            profit = position.calculate_profit()
            all_returns.append(profit)

        avg_return = np.mean(all_returns)
        std_dev = np.std(all_returns)
        if std_dev != 0:
            sharpe_ratio = (avg_return - risk_free_rate) / std_dev
        else:
            sharpe_ratio = np.nan
        return sharpe_ratio

    def calculate_win_rate(self):
        profitable_trades = sum([1 for trade in self.trade_history if trade['profit'] > 0])
        total_trades = len(self.trade_history)
        if total_trades > 0:
            win_rate = profitable_trades / total_trades * 100
        else:
            win_rate = np.nan
        return win_rate
    
    def calculate_max_drawdown(self):
        max_drawdown = 0
        max_value = 0
        for trade in self.trade_history:
            buy_date = trade['buy_date']
            sell_date = trade['sell_date']
            buy_price = trade['buy_price']
            sell_price = trade['sell_price']
            shares = trade['shares']
            symbol = trade['symbol']
            position = Position(symbol, buy_date, buy_price, shares)
            position.sell(sell_date, sell_price)
            value = position.calculate_profit()
            if value > max_value:
                max_value = value
            drawdown = (value - max_value) / max_value
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def calculate_sortino_ratio(self, risk_free_rate=0.02):
        all_returns = []
        for trade in self.trade_history:
            buy_date = trade['buy_date']
            sell_date = trade['sell_date']
            buy_price = trade['buy_price']
            sell_price = trade['sell_price']
            shares = trade['shares']
            symbol = trade['symbol']
            position = Position(symbol, buy_date, buy_price, shares)
            position.sell(sell_date, sell_price)
            profit = position.calculate_profit()
            all_returns.append(profit)

        avg_return = np.mean(all_returns)
        neg_returns = [r for r in all_returns if r < 0]
        std_dev = np.std(neg_returns)
        if std_dev != 0:
            sortino_ratio = (avg_return - risk_free_rate) / std_dev
        else:
            sortino_ratio = np.nan
        return sortino_ratio
    
    def calculate_profit_factor(self):
        profitable_trades = [trade['profit'] for trade in self.trade_history if trade['profit'] > 0]
        losing_trades = [trade['profit'] for trade in self.trade_history if trade['profit'] < 0]
        if sum(losing_trades) != 0:
            profit_factor = sum(profitable_trades) / abs(sum(losing_trades))
        else:
            profit_factor = np.nan
        return profit_factor
    
    def calculate_expectancy(self):
        profitable_trades = [trade['profit'] for trade in self.trade_history if trade['profit'] > 0]
        losing_trades = [trade['profit'] for trade in self.trade_history if trade['profit'] < 0]
        if len(profitable_trades) > 0 and len(losing_trades) > 0:
            expectancy = (sum(profitable_trades) / len(profitable_trades)) - (abs(sum(losing_trades)) / len(losing_trades))
        else:
            expectancy = np.nan
        return expectancy
    
    def calculate_r_squared(self):
        all_returns = []
        for trade in self.trade_history:
            buy_date = trade['buy_date']
            sell_date = trade['sell_date']
            buy_price = trade['buy_price']
            sell_price = trade['sell_price']
            shares = trade['shares']
            symbol = trade['symbol']
            position = Position(symbol, buy_date, buy_price, shares)
            position.sell(sell_date, sell_price)
            all_returns.append(position.calculate_profit())
        returns = pd.Series(all_returns)
        r_squared = returns.autocorr()
        return r_squared
    
    def calculate_metrics(self):
        self.sharpe_ratio = self.calculate_sharpe_ratio()
        self.profit_pct = ((self.capital - self.starting_capital) / self.starting_capital * 100).__round__(2)
        self.win_rate = self.calculate_win_rate()
        self.max_drawdown = self.calculate_max_drawdown()
        self.sortino_ratio = self.calculate_sortino_ratio()
        self.profit_factor = self.calculate_profit_factor()
        self.expectancy = self.calculate_expectancy()
        self.r_squared = self.calculate_r_squared()

    def get_metrics(self):
        return {
            'Sharpe Ratio': self.sharpe_ratio,
            'Profit Percent':  ((self.capital - self.starting_capital) / self.starting_capital * 100).__round__(2),
            'Win Rate': self.win_rate,
            'Max Drawdown': self.max_drawdown,
            'Sortino Ratio': self.sortino_ratio,
            'Profit Factor': self.profit_factor,
            'Expectancy': self.expectancy,
            'R Squared': self.r_squared
        }
    
    def get_total_profit(self):
        return 'Total Profit: $', self.total_profit.__round__(2)
    
    def get_total_profit_percent(self):
        return 'Total Profit : ', ((self.capital - self.starting_capital) / self.starting_capital * 100).__round__(2), '%'
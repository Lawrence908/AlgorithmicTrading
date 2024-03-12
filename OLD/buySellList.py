import sys
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ta
import pandas_datareader as pdr
import pytictoc as tt

# class Positions:
# 	def __init__(self):
# 		self.buy_dates = []
# 		self.sell_dates = []
# 		self.buy_prices = []
# 		self.sell_prices = []


class Signals:

	def __init__(self, symbol, spy_df, start = pd.to_datetime('today') - pd.DateOffset(months=12), end = pd.to_datetime('today')):
		self.spy_df = spy_df
		self.symbol = symbol
		self.start = start
		self.end = end
		self.stop_loss = 0.90
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
			self.plot_chart()

			self.recent_trade_df()
			self.trades_df()
			self.calc_buy_and_hold()
			self.plot_trades()

			self.profit = self.calc_profit()
			self.cumulative_profit = (self.profit + 1).prod() - 1
			self.max_drawdown = self.cumulative_profit.min()




	def calc_indicators(self):
		self.df['ma_20'] = self.df['Adj Close'].rolling(window=20).mean()
		self.df['vol'] = self.df['Adj Close'].rolling(window=20).std()
		self.df['upper_bb'] = self.df.ma_20 + (self.df.vol * 2)
		self.df['lower_bb'] = self.df.ma_20 - (self.df.vol * 2)
		self.df['rsi'] = ta.momentum.rsi(self.df['Adj Close'], window=6)


	def generate_signals(self):
		conditions = [(self.df.rsi < 30) & (self.df['Adj Close'] < self.df.lower_bb),
						(self.df.rsi > 70) & (self.df['Adj Close'] > self.df.upper_bb)]
		choices = ['Buy', 'Sell']
		self.df['signal'] = np.select(conditions, choices)
		self.df.dropna(inplace=True)
		self.df.signal = self.df.signal.shift(1)
		self.df['shifted_close'] = self.df['Adj Close'].shift(1)
		self.df['signal_strength'] = np.where(self.df['signal'] == 'Buy', self.df['Adj Close'] - self.df['lower_bb'], self.df['upper_bb'] - self.df['Adj Close'])

	def filter_signals_non_overlapping(self):
		self.position = False
		self.buy_dates, self.sell_dates = [], []
		self.buy_prices, self.sell_prices = [], []

		for index, row in self.df.iterrows():
			if not self.position and row.signal == 'Buy':
				self.buy_dates.append(index)
				self.buy_prices.append(row.Open)
				self.position = True


			# Add a stop loss condition
			if self.position and row.signal == 'Sell':
				self.sell_dates.append(index)
				self.sell_prices.append(row.Open)
				self.position = False

		self.buy_arr = self.df.loc[self.buy_dates].Open
		self.sell_arr = self.df.loc[self.sell_dates].Open


	def plot_chart(self):
		plt.figure(figsize=(20, 10))
		plt.plot(self.df['Adj Close'], label=self.symbol, alpha=1)
		plt.plot(self.df['ma_20'], label='20 Day Moving Average', alpha=0.8)
		plt.plot(self.df['upper_bb'], label='Upper Bollinger Band', alpha=0.55)
		plt.plot(self.df['lower_bb'], label='Lower Bollinger Band', alpha=0.50)
		plt.scatter(self.buy_arr.index, self.buy_arr.values, label='Buy Signal', marker='^', color='green')
		plt.scatter(self.sell_arr.index, self.sell_arr.values, label='Sell Signal', marker='v', color='red')
		for i in range(len(self.buy_arr)):
			plt.text(self.buy_arr.index[i], self.buy_arr.values[i], "      $" + str(round(self.buy_arr.values[i], 2)), fontsize=10, color='g')
		for i in range(len(self.sell_arr)):
			plt.text(self.sell_arr.index[i], self.sell_arr.values[i], "      $" + str(round(self.sell_arr.values[i], 2)), fontsize=10, color='r')
		plt.title(self.symbol + ' Recent Trades ' + str(self.start.date()) + ' to ' + str(self.end.date()))
		plt.legend(loc='upper left')
		plt.savefig('figures/trades/' + self.symbol + '.png')
		# plt.show()
		plt.close()


	def recent_trade_df(self):
		self.recent_trade_df = pd.DataFrame({})
		try:
			if len(self.buy_arr) > 0 and self.buy_arr.index[-1] > self.sell_arr.index[-1]:
				self.recent_trade_df = self.recent_trade_df._append({'Ticker': self.symbol, 'Trade': 'BUY', 'Date': self.buy_arr.index[-1], 'Price': self.buy_arr.values[-1], 'Signal Strength': self.df.loc[self.buy_arr.index[-1]]['signal_strength']}, ignore_index=True)
			if len(self.sell_arr) > 0 and self.buy_arr.index[-1] < self.sell_arr.index[-1]:
				self.recent_trade_df = self.recent_trade_df._append({'Ticker': self.symbol, 'Trade': 'SELL', 'Date': self.sell_arr.index[-1], 'Price': self.sell_arr.values[-1], 'Signal Strength': self.df.loc[self.sell_arr.index[-1]]['signal_strength']}, ignore_index=True)
		except:
			pass
		
		# if len(self.buy_arr) > 0:
		#     self.recent_trade_df = self.recent_trade_df._append({'Ticker': self.symbol, 'Trade': 'BUY', 'Date': self.buy_arr.index[-1], 'Price': self.buy_arr.values[-1],}, ignore_index=True)
		#     self.recent_trade_df['Signal Strength'] = self.df.loc[self.recent_trade_df['Date']]['signal_strength'].values
		# if len(self.sell_arr) > 0:
		#     self.recent_trade_df = self.recent_trade_df._append({'Ticker': self.symbol, 'Trade': 'SELL', 'Date': self.sell_arr.index[-1], 'Price': self.sell_arr.values[-1]}, ignore_index=True)
		#     self.recent_trade_df['Signal Strength'] = self.df.loc[self.recent_trade_df['Date']]['signal_strength'].values


	def trades_df(self):
		if len(self.buy_arr) != len(self.sell_arr):
			if len(self.buy_arr) > len(self.sell_arr):
				self.buy_arr = self.buy_arr[:-1]
			else:
				self.sell_arr = self.sell_arr[:-1]
		
		self.trades_df = pd.DataFrame({'Buy Date': self.buy_arr.index, 'Buy Price': self.buy_arr.values, 'Sell Date': self.sell_arr.index, 'Sell Price': self.sell_arr.values})
		# Add the ticker in the dataframe and place it next to the index
		self.trades_df.insert(0, 'Ticker', self.symbol)
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
		plt.plot(self.trades_df['Buy Date'], self.trades_df['Ticker Cum Profit %'], drawstyle="steps-post", label=self.symbol + ' algo cumulative %', alpha=1)
		for i in range(len(self.trades_df)):
			plt.text(self.trades_df['Buy Date'].iloc[i], self.trades_df['Ticker Cum Profit %'].iloc[i], "      " + str(round(self.trades_df['Profit %'].iloc[i], 2)) + "%", fontsize=10, color='black')
		

		plt.plot(self.spy_df['SPY Cumulative %'], drawstyle="steps-post", label='SPY cumulative %', alpha=0.8)

		plt.plot(self.df['Control Cumulative %'], drawstyle="steps-post", label=self.symbol + ' buy & hold cumulative %', alpha=0.8)

		plt.title(self.symbol + ' Algo vs. SPY vs. buy & hold ' + str(self.start.date()) + ' to ' + str(self.end.date()))
		plt.legend(loc='upper left')
		plt.savefig('figures/profit/' + self.symbol + '.png')
		# plt.show()
		plt.close()


	def calc_profit(self):
		if len(self.buy_arr) == 0 or len(self.sell_arr) == 0:
			return np.array([])  # Return an empty array if buy_arr or sell_arr is empty
		if self.buy_arr.index[-1] > self.sell_arr.index[-1]:
			self.buy_arr = self.buy_arr[:-1]
		return (self.sell_arr.values.sum() - self.buy_arr.values.sum())/self.buy_arr.values.sum()





if __name__ == "__main__":
	t = tt.TicToc()
	t.tic() # start timer

	# This is to avoid the SettingWithCopyWarning in pandas
	pd.options.mode.copy_on_write = True 

	# Read the filename provided in command line arg,
	# store the tickers in a list
	with open(sys.argv[1], 'r') as f:
		ticker_list = f.read().splitlines()

	# Download the SPY data
		spy_df = yf.download('SPY', start = pd.to_datetime('today') - pd.DateOffset(months=12), end = pd.to_datetime('today'), progress=False)
		spy_df['SPY'] = spy_df['Close'] - spy_df['Open']
		spy_df['SPY %'] = (spy_df['SPY'] / spy_df['Open']) * 100
		spy_df['SPY Cumulative %'] = spy_df['SPY %'].cumsum()

	# # Create a position object to store all trades for all tickers
	# positions = Positions()

	# Create a dataframe to store all trades for all tickers
	trades_df = pd.DataFrame()

	# Create a dataframe to store the most recent trade for all tickers
	recent_trade_df = pd.DataFrame()

	# Loop through the tickers and create a Signals object for each ticker
	for count, ticker in enumerate(ticker_list):
		SIG = Signals(ticker, spy_df)
		trades_df = trades_df._append(SIG.trades_df)
		try:
			recent_trade_df = recent_trade_df._append(SIG.recent_trade_df.iloc[-1])
		except:
			pass

		# SIG.df.to_csv('CSV/' + ticker + '.csv', index=False)

	# Sort trades_df by Ticker and Date descending and reset the index
	trades_df.sort_values(by=['Ticker', 'Buy Date'], ascending=[True, False], inplace=True)
	trades_df.reset_index(drop=True, inplace=True)

	# Sort recent_trades_df by Date descending and reset the index
	recent_trade_df.sort_values(by='Date', ascending=False, inplace=True)
	recent_trade_df.reset_index(drop=True, inplace=True)

	# Print the trades_df and recent_trade_df
	print(trades_df)
	print(recent_trade_df)

	# Save trades_df to a csv file
	trades_df.to_csv('CSV/trades.csv', index=False)

	#Save recent_trade_df to a csv file
	recent_trade_df.to_csv('CSV/recent_trades.csv', index=False)

	# Callculate the win rate
	win_rate = trades_df['Profitable'].value_counts(normalize=True)['Yes'] * 100
	print("Win Rate: ", win_rate, "%")

	# Calculate the cumulative profit
	cumulative_profit_pct = trades_df['Profit %'].sum()
	print("Cumulative Profit: ", cumulative_profit_pct, "%")


	t.toc() #elapsed time
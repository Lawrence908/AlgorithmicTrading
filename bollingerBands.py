import sys
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import pandas_ta as ta
import ta
import pandas_datareader as pdr



# Create a function to print space between data output
# Default is * characters and multiple of 50
def print_space(char='*', num=50):
    print(char * num)

if __name__ == "__main__":
    # I want to read a list of tickers from a file, and store them in a list
    # I will then use the list to download the stock data by iterating over the list
    # I will use this list to plot the moving averages and indicate when a buy or sell signal is generated

    # This is to avoid the SettingWithCopyWarning in pandas
    pd.options.mode.copy_on_write = True 

    # Read the filename provided in command line arg,
    # store the tickers in a list
    with open(sys.argv[1], 'r') as f:
        ticker_list = f.read().splitlines()

    for count, ticker in enumerate(ticker_list):

        if '.' in ticker:
            ticker = ticker.replace('.', '-')

        # print_space()
        # print("Analyzing Stock Ticker: ", ticker)

        # # Start date = todays date offset by 6 months
        # start_timestamp = pd.to_datetime('today') - pd.DateOffset(months=60)
        # start:str = start_timestamp.strftime('%Y-%m-%d')
        # End date = todays date
        # end_timestamp = pd.to_datetime('today')
        # end:str = end_timestamp.strftime('%Y-%m-%d')
        start = '2022-01-31'
        end = '2024-01-31'

        # Download the stock data without it printing download status to the console
        df = yf.download(ticker, start, end, progress=False)

        # #if the dataframe is empty, then skip the rest of the code and go to the next ticker
        if df.empty:
            continue

        # df.drop(columns=['High', 'Low', 'Close', 'Volume'], inplace=True)

        # Add a days counter to the stock data
        # And put it to the first column
        day = np.arange(1, len(df) + 1)
        df['Day'] = day
        df = df[['Day'] + [col for col in df.columns if col != 'Day']]

        # Calculate the moving average
        # And add it to the dataframe
        df['ma_20'] = df['Adj Close'].rolling(window=20).mean()
        df['vol'] = df['Adj Close'].rolling(window=20).std()
        df['upper_bb'] = df.ma_20 + (df.vol * 2)
        df['lower_bb'] = df.ma_20 - (df.vol * 2)

        # Calculate the RSI
        df['rsi'] = ta.momentum.rsi(df['Adj Close'], window=6)

        # Create a list of conditions to check for buy or sell signals
        conditions = [(df.rsi < 30) & (df['Adj Close'] < df.lower_bb),
                        (df.rsi > 70) & (df['Adj Close'] > df.upper_bb)]
        
        # Create a list of choices to select from
        choices = ['Buy', 'Sell']

        # Create a new column to store the buy or sell signals based on confitions and choices
        # df['signal'] = np.select(conditions, choices, default='Hold')
        df['signal'] = np.select(conditions, choices)

        #Drop the NaN values
        df.dropna(inplace=True)
        
        # Shift the signal column by 1 to avoid look ahead bias
        df.signal = df.signal.shift(1)

        # Create a new column to store the shifted close price
        df['shifted_close'] = df['Adj Close'].shift(1)

        position = False
        buy_dates, sell_dates = [], []
        buy_prices, sell_prices = [], []

        # Iterate over the rows of the dataframe
        for index, row in df.iterrows():
            if not position and row.signal == 'Buy':
                buy_dates.append(index)
                buy_prices.append(row.Open)
                position = True

            if position:
                if row.signal == 'Sell'  or row.shifted_close < 0.95 * buy_prices[-1]:
                    sell_dates.append(index)
                    sell_prices.append(row.Open)
                    position = False

        # Track the gain and loss of each buy and sell period to cumulatively calculate the return of the strategy
        # Plot each gain and loss on the chart
                    
        # Calculate and print the returns of the strategy
        # Later we should report this through streamlit or some other web framework with a nice UI showing best and worst performing stocks
        print(ticker, "Gain/Loss:", "{:.2%}".format((pd.Series([(sell - buy) / buy for sell, buy in zip(sell_prices, buy_prices)]) + 1).prod() - 1))


        plt.figure(figsize=(20,10))
        plt.plot(df['Adj Close'], label=ticker, alpha=1)
        plt.plot(df['ma_20'], label='20 Day Moving Average', alpha=0.80)
        plt.plot(df['upper_bb'], label='Upper Bollinger Band', alpha=0.55)
        plt.plot(df['lower_bb'], label='Lower Bollinger Band', alpha=0.50)
        plt.scatter(df.loc[buy_dates].index, df.loc[buy_dates]['Adj Close'], marker= '^', color= 'g')
        plt.scatter(df.loc[sell_dates].index, df.loc[sell_dates]['Adj Close'], marker= 'v', color= 'r')
        for i in range(len(buy_dates)):
            plt.text(buy_dates[i], df.loc[buy_dates[i]]['Adj Close'], "          $" + str(round(buy_prices[i], 2)), fontsize=10, color='g')
        for i in range(len(sell_dates)):
            plt.text(sell_dates[i], df.loc[sell_dates[i]]['Adj Close'], "          $" + str(round(sell_prices[i], 2)), fontsize=10, color='r')
        
        # Another line on the chart will track the cumulative gain and loss of the strategy over the entire period
        # I want to see how the strategy is performing over time, show the gain loss % at each trade on this line
        # This will use a separate y axis on the right side of the chart
        plt.twinx()
        plt.plot(pd.Series([(sell - buy) / buy for sell, buy in zip(sell_prices, buy_prices)]).cumsum(), label='Cumulative Gain/Loss', color='b', alpha=0.5)
        # plt.axhline(0, color='black', linestyle='--', alpha=0.5)


        # There is a random piece of data in the 1970s that is causing the chart to be squished
        # This line will remove that data from the chart
        plt.xlim(pd.to_datetime(start), pd.to_datetime(end))

        plt.title('Bollinger Bands for ' + ticker + ' from ' + start + ' to ' + end)
        # plt.xlabel('Date')
        # plt.ylabel('Price')
        plt.legend()

        plt.savefig('figures/bollingerBands/' + str(count + 1) + " - " + ticker + end + '.png')



        #Close the plot
        plt.close()
import sys
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta
# import ta
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

        # print_space()
        # print("Analyzing Stock Ticker: ", ticker)

        # if there is a '.' in one of the characters ticker str, replace it with a '-' to avoid file naming issues
        if '.' in ticker:
            ticker = ticker.replace('.','-')

        # # Start date = todays date offset by 6 months
        start_timestamp = pd.to_datetime('today') - pd.DateOffset(months=9)
        start:str = start_timestamp.strftime('%Y-%m-%d')
        # End date = todays date
        end_timestamp = pd.to_datetime('today')
        end:str = end_timestamp.strftime('%Y-%m-%d')

        # Download the stock data without it printing download status to the console
        df = yf.download(ticker, start, end, progress=False)

        #if the dataframe is empty, then skip the rest of the code and go to the next ticker
        if df.empty:
            continue

        # df.drop(columns=['High', 'Low', 'Close', 'Volume'], inplace=True)

        # Add a days counter to the stock data
        # And put it to the first column
        day = np.arange(1, len(df) + 1)
        df['Day'] = day
        df = df[['Day'] + [col for col in df.columns if col != 'Day']]

        # Calculate the simple moving average
        # And add it to the dataframe
            # the below command would be great but it uses 'Close' instead of 'Adj Close' price
            # df.ta.sma(length=5, append=True)
        df['SMA_5'] = ta.sma(df['Adj Close'],5)
        df['SMA_10'] = ta.sma(df['Adj Close'],10)
        df['SMA_20'] = ta.sma(df['Adj Close'],20)
        df['SMA_50'] = ta.sma(df['Adj Close'],50)
        
        # Calculate the daily log returns based on the adjusted close price
        # And add it to the dataframe
        log_returns = np.log(1 + df['Adj Close'].pct_change())
        df['Returns'] = log_returns

        # Calculate the cumulative log returns with exponential
        # And add it to the dataframe
        cum_log_returns = np.exp(np.log(1 + df['Adj Close'].pct_change()).cumsum()) - 1
        df['Cum_Returns'] = cum_log_returns

        #Drop the NaN values
        df.dropna(inplace=True)

        # Calculate the buy or sell signal based on the moving averages
        # And add it to the dataframe
        df['Buy_Sell_5_10'] = np.where(df['SMA_5'] > df['SMA_10'], 1, 0)
        df['Buy_Sell_5_10'] = np.where(df['SMA_5'] < df['SMA_10'], -1, df['Buy_Sell_5_10'])

        df['Buy_Sell_10_20'] = np.where(df['SMA_10'] > df['SMA_20'], 1, 0)
        df['Buy_Sell_10_20'] = np.where(df['SMA_10'] < df['SMA_20'], -1, df['Buy_Sell_10_20'])

        df['Buy_Sell_20_50'] = np.where(df['SMA_20'] > df['SMA_50'], 1, 0)
        df['Buy_Sell_20_50'] = np.where(df['SMA_20'] < df['SMA_50'], -1, df['Buy_Sell_20_50'])

        # Now I want to plot the moving averages and an entry based on the buy or sell signal
        # I will use the matplotlib library to plot the data
        # I want the Buy signal to be a green arrow pointing up
        # I want the Sell signal to be a red arrow pointing down
        plt.figure(figsize=(20,10))
        plt.title('Moving Averages for ' + ticker + ' from ' + start + ' to ' + end)
        plt.plot(df['Adj Close'], label=ticker, alpha=1)
        plt.plot(df['SMA_5'], label='SMA_5', alpha=0.80)
        plt.plot(df['SMA_10'], label='SMA_10', alpha=0.75)
        plt.plot(df['SMA_20'], label='SMA_20', alpha=0.70)
        plt.plot(df['SMA_50'], label='SMA_50', alpha=0.65)
        plt.scatter(df[df['Buy_Sell_5_10'] == 1].index, df['SMA_5'][df['Buy_Sell_5_10'] == 1], marker='^', color='g')
        plt.scatter(df[df['Buy_Sell_5_10'] == -1].index, df['SMA_5'][df['Buy_Sell_5_10'] == -1], marker='v', color='r')
        plt.scatter(df[df['Buy_Sell_10_20'] == 1].index, df['SMA_10'][df['Buy_Sell_10_20'] == 1], marker='^', color='g')
        plt.scatter(df[df['Buy_Sell_10_20'] == -1].index, df['SMA_10'][df['Buy_Sell_10_20'] == -1], marker='v', color='r')
        plt.scatter(df[df['Buy_Sell_20_50'] == 1].index, df['SMA_20'][df['Buy_Sell_20_50'] == 1], marker='^', color='g')
        plt.scatter(df[df['Buy_Sell_20_50'] == -1].index, df['SMA_20'][df['Buy_Sell_20_50'] == -1], marker='v', color='r')
        plt.legend()



        # if the Buy_Sell_5_10 is 1, the Buy_Sell_10_20 is 1, and the Buy_Sell_20_50 is 1, then the entry is 3
        # if the Buy_Sell_5_10 is -1, the Buy_Sell_10_20 is -1, and the Buy_Sell_20_50 is -1, then the entry is -3
        # if the Buy_Sell_5_10 is 1, the Buy_Sell_10_20 is 1, and the Buy_Sell_20_50 is -1, then the entry is 2
        # if the Buy_Sell_5_10 is -1, the Buy_Sell_10_20 is -1, and the Buy_Sell_20_50 is 1, then the entry is -2
        # make an entry column that is the sum of the Buy_Sell_5_10, Buy_Sell_10_20, and Buy_Sell_20_50
        df['entry'] = df['Buy_Sell_5_10'] + df['Buy_Sell_10_20'] + df['Buy_Sell_20_50']

        # if the entry is 3 for the last 10 rows, then print a buy signal
        if all(df['entry'].iloc[-10:] == 3):
            print("Buy: ", ticker)
            # Save the plot to a file in the figures directory with the ticker name and the moving averages
            plt.savefig('figures/shortTerm/' + str(count+ 1) + " - " + ticker + "_SMAs_"+ end + '.png')

        # if the entry is -3 for the last 10 rows, then print a sell signal
        if all(df['entry'].iloc[-10:] == -3):
            print("Sell: ", ticker)
            # Save the plot to a file in the figures directory with the ticker name and the moving averages
            plt.savefig('figures/shortTerm/' + str(count + 1) + " - " + ticker + "_SMAs_"+ end + '.png')

        # Print the last 5 rows of the dataframe
        # print(df.tail())
            
        #Close the plot
        plt.close()
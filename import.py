import sys
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr




# Create a function to print space between data output
# Default is * characters and multiple of 50
def print_space(char='*', num=50):
    print(char * num)

# Create a function to get stock data using yfinance
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start = start, end = end)
    df.drop(columns=['High', 'Low', 'Close', 'Volume'], inplace=True)
    return df

# Add a days counter to the stock data
# And put it to the first column
def add_days_counter(df):
    day = np.arange(1, len(df) + 1)
    df['Day'] = day
    df = df[['Day'] + [col for col in df.columns if col != 'Day']]
    return df

# Caluclate the daily returns
def calculate_daily_returns(df):
    returns = df['Adj Close'].pct_change()
    return returns

# # Calculate the cumulative returns
# def calculate_cumulative_returns(df):
#     cumulative_return = df["Adj Close"] / df["Adj Close"].shift(3) - 1
#     return cumulative_return


# cumulated_return = (1 + returns).cumprod() - 1
# print(cumulated_return)

# Calculate the log returns
def calculate_log_returns(df):
    log_returns = np.log(1 + df['Adj Close'].pct_change())
    return log_returns

# Caclulate the cumulative log returns with exponential
# And add it to the dataframe

def calculate_cum_log_returns(df):
    cum_log_returns = np.exp(np.log(1 + df['Adj Close'].pct_change()).cumsum()) - 1
    df['Cum_Log_Returns'] = cum_log_returns
    return df


# sum_returns = log_returns.sum()
# print(sum_returns)
    
# Calculate the simple moving average
# Add the column to the dataframe and return the dataframe
def calculate_moving_avg(df, window):
    moving_avg = df['Adj Close'].rolling(window=window).mean().shift(1)
    df['SMA_' + str(window)] = moving_avg
    return df

# Calculate the buy or sell signal based on the moving averages
# Add the column to the dataframe and return the dataframe
def calculate_buy_sell_signal(df, sma1, sma2):
    df[sma1 + '_' + sma2 + '_signal'] = np.where(df['SMA_' + str(sma1)] > df['SMA_' + str(sma2)], 1, 0) 
    df[sma1 + '_' + sma2 + '_signal'] = np.where(df['SMA_' + str(sma1)] < df['SMA_' + str(sma2)], -1, df[sma1 + '_' + sma2 + '_signal']) 
    return df


# Calculate the instantaneous return
def calculate_returns(df):
    returns = np.log(df['Adj Close']).diff()
    df['Returns'] = returns
    return df

# Calculate the system return
def calculate_system_return(df):
    system_return = df['Cum_Log_Returns'] * df['9_21_signal'].shift(1)
    df['System_Return'] = system_return
    return df

# Now I want to compare these moving avgs in a plot
def plot_moving_avg(df, ticker, sma1, sma2, start, end):
    plt.figure(figsize=(20,10))
    plt.title('Moving Averages for ' + ticker + ' from ' + start + ' to ' + end)
    plt.plot(df['Adj Close'], label=ticker)
    plt.plot(df['SMA_' + str(sma1)], label='SMA_' + str(sma1))
    plt.plot(df['SMA_' + str(sma2)], label='SMA_' + str(sma2))
    # Create a green arrow up for entry == 2
    plt.plot(df[df['entry'] == 2].index, df['SMA_' + str(sma1)][df['entry'] == 2], '^', markersize=10, color='g')
    # Create a red arrow down for entry == -2
    plt.plot(df[df['entry'] == -2].index, df['SMA_' + str(sma1)][df['entry'] == -2], 'v', markersize=10, color='r')
    plt.legend()
    return plt

# # Create a function to calculate the moving average crossover strategy
# def moving_avg_crossover_strategy(df, ticker, start, end):
#     df = get_stock_data(ticker, start, end)
#     df = add_days_counter(df)
#     df = calculate_cum_log_returns(df)
#     df = calculate_moving_avg(df, 9)
#     df = calculate_moving_avg(df, 21)
#     df = calculate_buy_sell_signal(df, '9', '21')
#     df.dropna(inplace=True)
#     df = calculate_instantaneous_returns(df)
#     df = calculate_system_return(df)
#     df['entry'] = df['9_21_signal'].diff()
#     plot = plot_moving_avg(df, ticker, 9, 21, start, end)
#     plt.savefig(ticker + "_moving_avg_"+ start + '_to_' + end + '.png')

#     return df

# Plot the instantaneous returns, system returns and the stock price
def plot_returns(df, ticker, start, end):
    plt.figure(figsize=(20,10))
    plt.title('Returns for ' + ticker + ' from ' + start + ' to ' + end)
    plt.plot(np.exp(df['Returns']).cumprod(), label='Returns')
    plt.plot(np.exp(df['System_Return']).cumprod(), label='System Returns')
    plt.plot(df['Adj Close'], label=ticker)
    plt.legend()
    return plt



if __name__ == "__main__":
    ticker = sys.argv[1]

    start = '2022-01-01' 
    end = '2024-02-01'

    df = get_stock_data(ticker, start, end)

    df = add_days_counter(df)

    # daily_returns = calculate_daily_returns(df)
    # cumulative_returns = calculate_cumulative_returns(df)
    # log_returns = calculate_log_returns(df)
    df = calculate_cum_log_returns(df)

    # Calculate the moving averages
    df = calculate_moving_avg(df, 9)
    df = calculate_moving_avg(df, 14)
    df = calculate_moving_avg(df, 21)

    df = calculate_moving_avg(df, 30)
    df = calculate_moving_avg(df, 60)
    df = calculate_moving_avg(df, 90)

    df = calculate_moving_avg(df, 100)
    df = calculate_moving_avg(df, 150)
    df = calculate_moving_avg(df, 200)

    # Calculate buy sell signal using the moving averages
    df = calculate_buy_sell_signal(df, '9', '21')
    # df = calculate_buy_sell_signal(df, '30', '90')
    # df = calculate_buy_sell_signal(df, '100', '200')

    #Drop the NaN values
    df.dropna(inplace=True)

    df = calculate_returns(df)
    df = calculate_system_return(df)
    df['entry'] = df['9_21_signal'].diff()

    print_space()
    print(df.head())


    



    print_space()

    print("Stock Ticker: ", ticker)
    print(df[245:255])

    #Call plot_moving_avg function to plot the moving averages and assign it to a variable
    plot = plot_moving_avg(df, ticker, 9, 21, start, end)
    plt.savefig(ticker + "9-21_moving_avg_"+ start + '_to_' + end + '.png')

    plot = plot_moving_avg(df, ticker, 30, 90, start, end)
    plt.savefig(ticker + "30-90_moving_avg_"+ start + '_to_' + end + '.png')    
    
    plot = plot_moving_avg(df, ticker, 100, 200, start, end)
    plt.savefig(ticker + "100-200_moving_avg_"+ start + '_to_' + end + '.png')



    # Call plot_returns function and print the plot
    plot_returns = plot_returns(df, ticker, start, end)
    plt.savefig(ticker + "_returns_"+ start + '_to_' + end + '.png')


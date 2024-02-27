import sys
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ta
import pandas_datareader as pdr
import backTest as bt




if __name__ == "__main__":
    t = TicToc()
    t.tic() # start timer

    # This is to avoid the SettingWithCopyWarning in pandas
    pd.options.mode.copy_on_write = True 

    # Read the filename provided in command line arg,
    # store the tickers in a list
    with open(sys.argv[1], 'r') as f:
        ticker_list = f.read().splitlines()

        # Create a dataframe to store all trades for all tickers
        trades_df = pd.DataFrame()

    for count, ticker in enumerate(ticker_list):
        BT = bt.Backtest(ticker)
        # trades_df = trades_df.append(BT.trades_df)
        trades_df = trades_df.append(BT.trades_df, ignore_index=True)

    trades_df.to_csv('trades.csv', index=False)




    t.toc() #elapsed time
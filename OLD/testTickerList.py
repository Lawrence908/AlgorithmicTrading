import sys
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr

if __name__ == "__main__":
    # I want to read a list of tickers from a file, and store them in a list
    # I will then use the list to download the stock data by iterating over the list
    # I will use this list to plot the moving averages and indicate when a buy or sell signal is generated

    # Read the file and store the tickers in a list
    # with open('tickerList.txt', 'r') as f:
    with open(sys.argv[1], 'r') as f:
        ticker_list = f.read().splitlines()

    for ticker in ticker_list:
        print("Stock Ticker: ", ticker)
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


def get_tickers500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers500 = []
    for row in table.find_all('tr')[1:]:
        ticker = row.find('td').text.strip()
        tickers500.append(ticker)
    return tickers500

def add_z_score(data: pd.DataFrame, frame: str = 'Adj Close', window1 = 20) -> pd.DataFrame:
    data['Z Score ' + frame] = (data[frame] - data[frame].rolling(window= window1).mean()) / data[frame].rolling(window= window1).std()
    data.dropna(inplace=True)
    return data

def get_z_score(data: pd.DataFrame, frame: str = 'Adj Close', window1 = 20) -> pd.DataFrame:
    data['Z Score ' + frame] = (data[frame] - data[frame].rolling(window= window1).mean()) / data[frame].rolling(window= window1).std()
    data.dropna(inplace=True)
    return data['Z Score ' + frame]

if __name__ == '__main__':
    tickers500 = get_tickers500()
    stock_df = pd.DataFrame()
    tradingstrategy1_df = pd.DataFrame()

    for ticker in tickers500[1:500]:
        stock = Ticker(ticker, start='2022-01-31', end='2024-01-31')
        stock_df = stock_df._append(stock.df)
        techA = TechnicalAnalysis(stock)
        tradingS = TradingStrategy1(techA)
        tradingstrategy1_df = tradingstrategy1_df._append(tradingS.trades_df)



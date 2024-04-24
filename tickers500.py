import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import random


url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
table = soup.find('table', {'id': 'constituents'})
tickers500 = []
for row in table.find_all('tr')[1:]:
    ticker = row.find('td').text.strip()
    tickers500.append(ticker)

# I want to create a class that will:
# - get the tickers of the S&P 500 companies
# - iterate over the tickers and get the data of each ticker
# - store the data in a pandas DataFrame, save it as a csv file
# - another function to update the data of the tickers before returning the data

class Tickers500:
    def __init__(self):
        self.tickers = self.get_tickers()
        
    def get_tickers(self):
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        tickers500 = []
        for row in table.find_all('tr')[1:]:
            ticker = row.find('td').text.strip()
            tickers500.append(ticker)
        return tickers500

    # remove progress bar
    def get_data(self):
        for ticker in self.tickers[1:15]:
            if '.' in ticker:
                ticker = ticker.replace('.', '-')
            data = yf.download(ticker, progress=False)
            data.to_csv("TickerData/"f'{ticker}.csv')

    def update_data(self):
        for ticker in self.tickers[1:15]:
            recent_date = pd.read_csv("TickerData/"f'{ticker}.csv').tail(1).index[0]
            recent_date = pd.read_csv("TickerData/"f'{ticker}.csv').iloc[recent_date].Date
            recent_date = pd.to_datetime(recent_date)
            recent_date = recent_date + pd.DateOffset(days=1)
            data = yf.download(ticker, start=recent_date, progress=False)
            data.to_csv("TickerData/"f'{ticker}.csv', mode='a', header=False)

    def load_ticker_data_to_df(self, ticker, start_date=None, end_date=None):
        data = pd.read_csv("TickerData/"f'{ticker}.csv')
        if start_date and end_date:
            data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
        if start_date:
            data = data[data['Date'] >= start_date]
        return data
    
    def get_random_ticker(self):
        return random.choice(self.tickers)
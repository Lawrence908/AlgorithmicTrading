import requests
from bs4 import BeautifulSoup

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
table = soup.find('table', {'id': 'constituents'})
tickers500 = []
for row in table.find_all('tr')[1:]:
    ticker = row.find('td').text.strip()
    tickers500.append(ticker)
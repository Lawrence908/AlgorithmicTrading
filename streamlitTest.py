import streamlit as st
import pandas as pd
from tickerTA import Ticker, TechnicalAnalysis
from tradingStrategy1 import TradingStrategy1
from recommendations import Recommendations
from tickers500 import tickers500

symbols = tickers500
st.title('Stock Analysis App')

#symbol selection
selected_symbol = st.sidebar.selectbox('Select a symbol', symbols)
ticker = Ticker(selected_symbol)
tech_analysis = TechnicalAnalysis(ticker)
tradingS = TradingStrategy1(tech_analysis)

#display basic info about the selected symbol
st.header(f'Analysis for {selected_symbol}')
st.subheader('Basic Info')
st.write(f"Company Name: {ticker.info['shortName']}")
st.write(f"Industry: {ticker.info['industry']}")
st.write(f"Sector: {ticker.info['sector']}")
st.write(f"Country: {ticker.info['country']}")
st.write(f"Exchange: {ticker.info['exchange']}")

#dropdowns
more_info = st.checkbox("More Info")
show_financials = st.checkbox("Show Financials")
show_ratios = st.checkbox("Show Ratios")
show_piotroski = st.checkbox("Show Piotroski F-Score")

if more_info:
    for key, value in ticker.info.items():
        st.write(f"{key}: {value}")

if show_financials:
    st.subheader('Financials')
    st.write("The financials are based on the latest available data.")
    financials_dict = ticker.financials
    last_key = list(financials_dict.keys())[-1]
    last_value = financials_dict[last_key]
    st.write(f"Date: {last_key}")
    for key, value in last_value.items():
        st.write(f"{key}: {value}")

if show_ratios:
    st.subheader('Ratios')
    st.write("The ratios are calculated based on the financial statements of the company.")
    ratios_dict = ticker.get_ratios()
    for key, value in ratios_dict.items():
        st.write(f"{key}: {value}")

if show_piotroski:
    st.subheader('Piotroski F-Score')
    st.write(f"The Piotroski F-Score is a number between 0 and 9 that reflects nine criteria used to determine the strength of a company's financial position.")
    st.write(f"A higher score indicates a stronger financial position.")
    st.write(f"The Piotroski F-Score for {selected_symbol}:")
    for key, value in ticker.piotroski.items():
        st.write(f"{key}: {value}")

#Ticker
st.subheader('Stock Price')
st.write("The stock price is plotted on the chart below.")
st.write("The adjusted closing price is used for the analysis.")
st.line_chart(tech_analysis.df_ta['Adj Close'])

#technical analysis
st.subheader('Technical Analysis')
st.write("Bollinger Bands:")
st.write("The Bollinger Bands are a volatility indicator that consists of a simple moving average and two standard deviations plotted above and below the moving average.")
st.write("The Bollinger Band High is the upper band, and the Bollinger Band Low is the lower band.")
st.write("The stock price is plotted along with the Bollinger Bands.")
st.line_chart(tech_analysis.df_ta[['Adj Close', 'BollingerBandHigh', 'BollingerBandLow']])

st.write("MACD (Moving Average Convergence Divergence):")
st.write("The MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a stock's price.")
st.line_chart(tech_analysis.df_ta[['MACD', 'MACD Signal']])


st.write("Short Term Moving Averages:")
st.write("The short term moving averages are plotted on the chart below.")
st.write("The stock price is plotted along with the short term moving averages.")
st.line_chart(tech_analysis.df_short_sma[['Adj Close', 'SMA_10', 'SMA_20', 'SMA_30']])

st.write("Long Term Moving Averages:")
st.write("The long term moving averages are plotted on the chart below.")
st.write("The stock price is plotted along with the long term moving averages.")
st.line_chart(tech_analysis.df_long_sma[['Adj Close', 'SMA_50', 'SMA_100', 'SMA_200']])

#Trading Strategy
st.subheader('Trading Strategy')
st.write("The trading strategy is based on the bollinger bands and the RSI (Relative Strength Index).")
st.write("The strategy generates buy and sell signals based on the following rules:")
st.write("1. Buy when the stock price crosses below the lower Bollinger Band and the RSI is below 30.")
st.write("2. Sell when the stock price crosses above the upper Bollinger Band and the RSI is above 70.")
st.write("The signals are plotted on the chart below.")
st.write("The green arrows indicate buy signals, and the red arrows indicate sell signals.")
st.pyplot(tradingS.plot_trades())

st.write("The trading strategy generated the following profits:")
st.pyplot(tradingS.plot_profit())
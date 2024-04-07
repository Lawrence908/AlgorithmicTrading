'''
This streamlit file does technical analysis of a selected stock which includes
MACD, MACD signal, short terma nd long term moving averages
This file gets symbols from tickers500 and methods from TickerTA and auto correlation plot
NOTE: autcorrelation plot is'nt working 

'''



import streamlit as st
import pandas as pd
from tickerTA import Ticker, TechnicalAnalysis
from tickers500 import tickers500

symbols = tickers500
st.title('Stock Analysis App')

#symbol selection
selected_symbol = st.sidebar.selectbox('Select a symbol', symbols)
ticker = Ticker(selected_symbol)

#display basic info about the selected symbol
st.header(f'Analysis for {selected_symbol}')
st.subheader('Basic Info')
st.write(f"Company Name: {ticker.info['shortName']}")
st.write(f"Industry: {ticker.info['industry']}")
st.write(f"Sector: {ticker.info['sector']}")
st.write(f"Country: {ticker.info['country']}")
st.write(f"Exchange: {ticker.info['exchange']}")

#dropdowns
show_financials = st.checkbox("Show Financials")
show_ratios = st.checkbox("Show Ratios")

if show_financials:
    st.subheader('Financials')
    financials_df = pd.DataFrame(ticker.financials.items(), columns=['Metric', 'Value'])
    financials_df.set_index('Metric', inplace=True)
    st.bar_chart(financials_df)

if show_ratios:
    st.subheader('Ratios')
    ratios_df = pd.DataFrame(ticker.ratios.items(), columns=['Ratio', 'Value'])
    ratios_df.set_index('Ratio', inplace=True)
    st.bar_chart(ratios_df)

#technical analysis
st.subheader('Technical Analysis')
tech_analysis = TechnicalAnalysis(ticker)
st.write("Bollinger Bands:")
st.line_chart(tech_analysis.df_ta[['BollingerBandHigh', 'BollingerBandLow']])
st.write("MACD (Moving Average Convergence Divergence):")
st.write("The MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a stock's price.")
st.line_chart(tech_analysis.df_ta[['MACD', 'MACD Signal']])
st.write("The MACD Signal line is a 9-day EMA (Exponential Moving Average) of the MACD line.")
st.write("When the MACD crosses above the MACD Signal, it may signal a bullish trend.")
st.write("When the MACD crosses below the MACD Signal, it may signal a bearish trend.")
st.write("Short Term Moving Averages:")
st.line_chart(tech_analysis.df_short_sma[['SMA_10', 'SMA_20', 'SMA_30']])
st.write("Long Term Moving Averages:")
st.line_chart(tech_analysis.df_long_sma[['SMA_50', 'SMA_100', 'SMA_200']])

#autocorrelation plot
st.subheader('Autocorrelation Plot')
tech_analysis.autocorrelation_plot()
st.pyplot()

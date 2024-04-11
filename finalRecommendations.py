import streamlit as st
import pandas as pd
from tickerTA import Ticker, TechnicalAnalysis
from tradingStrategy1 import TradingStrategy1
from recommendations import Recommendations
from tickers500 import tickers500

st.title('Algorithmic Trading App Recommendations')

#Recommendations
st.subheader("Today's Recommendations")
st.write("The recommendations are based on the trading strategy.")
rec = Recommendations()
st.write("Buy Recommendations:")
st.write(rec.buy_recommendations())


st.write("Sell Recommendations:")
st.write(rec.sell_recommendations())
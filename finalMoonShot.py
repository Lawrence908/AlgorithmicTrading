import pandas as pd
from tickers500 import tickers500
from tickerTA import Ticker
from tickerTA import TechnicalAnalysis
from tradingStrategy1 import TradingStrategy1
from moonShot2 import MoonShot2
from recommendations import Recommendations
import streamlit as st

# tradingstrategy1_df = pd.DataFrame()
# all_signals_df = pd.DataFrame()
buytable_df = pd.DataFrame()

for ticker in tickers500[0:50]:
        stock = Ticker(ticker, start='2022-01-31', end='2024-01-31')
        techA = TechnicalAnalysis(stock)
        tradingS = TradingStrategy1(techA, 'FinalPrototype')
        # all_signals_df = all_signals_df._append(tradingS.all_signals_df)
        # tradingstrategy1_df = tradingstrategy1_df._append(tradingS.trades_df)
        buytable_df = tradingS.get_buytable()._append(buytable_df)


moonShot2 = MoonShot2(buytable_df)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('MoonShot2 Prototype')

st.header('AI Training and Testing')

st.write('The AI model is trained on the buy table data of the 500 S&P companies.')
st.write('Here we see the loss, accuracy, and ROC curve of the model as it was trained in each epoch.')
st.pyplot(moonShot2.plot_loss())
st.pyplot(moonShot2.plot_accuracy())
st.pyplot(moonShot2.plot_roc_curve())

st.header('AI Results')
st.write('The following results represent the AI models training and testing on the buy table data of the 500 S&P companies, when the tradingStrategy1 class was used.')
st.write(moonShot2.output_results())

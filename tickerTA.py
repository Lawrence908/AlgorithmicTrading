import sys
import yfinance as yf
from pandas.plotting import autocorrelation_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

# This is to avoid the SettingWithCopyWarning in pandas
pd.options.mode.copy_on_write = True 

class Ticker:
    def __init__(self, symbol, start=pd.to_datetime('today') - pd.DateOffset(months=24), end=pd.to_datetime('today')):
        self.symbol = symbol
        self.start = start
        self.end = end
        if '.' in self.symbol:
            self.symbol = self.symbol.replace('.', '-')
        self.df = yf.download(self.symbol, self.start, self.end, progress=False)
        if self.df.empty:
            print("No data found for ", self.symbol)
            sys.exit(1)
        else:
            self.df = self.df.round(2)
            self.Ticker = yf.Ticker(self.symbol)
            self.actions = self.Ticker.get_actions()
            self.info = self.Ticker.get_info()
            self.financials = self.Ticker.get_financials()
            self.balance_sheet = self.Ticker.get_balance_sheet()
            self.income_statement = self.Ticker.get_income_stmt()
            self.cashflow_statement = self.Ticker.get_cashflow()
            self.calendar = self.Ticker.get_calendar()
            try:
                self.inst_holders = self.Ticker.get_institutional_holders()
            except:
                self.inst_holders = None
            # try:
            #     self.news = self.Ticker.get_news()
            # except:
            #     self.news = None
            
            try:
                if self.symbol != 'FOX' or self.symbol != 'NWS':
                    self.recommendations = self.Ticker.get_recommendations()
                else:
                    self.recommendations = None
            except:
                self.recommendations = None
            try:
                self.analysis = self.Ticker.get_analysis()
            except:
                self.analysis = None
            try:
                self.sustainability = self.Ticker.get_sustainability()
            except:
                self.sustainability = None
            try:
                self.profitability = self.get_profitability()
            except:
                self.profitability = None
            try:
                self.leverage = self.get_leverage()
            except:
                self.leverage = None
            try:
                self.operating_efficiency = self.get_operating_efficiency()
            except:
                self.operating_efficiency = None
            try:
                self.piotroski = self.get_piotroski()
            except:
                self.piotroski = None
            try:
                self.altman_z_score = self.get_altman_z_score()
            except:
                self.altman_z_score = None
            try:
                self.ratios = self.get_ratios()
                self.df['P/E Ratio'] = self.ratios['price_earnings_ratio']
                self.df['Fwd P/E Ratio'] = self.ratios['projected_price_earnings_ratio']
                self.df['P/B Ratio'] = self.ratios['profit_book_ratio']
            except:
                self.ratios = None
            self.df.insert(0, 'Ticker', self.symbol)
    
    
    def __str__(self):
        return self.symbol

    def __repr__(self):
        return self.symbol
    
    def get_profitability(self) -> dict:
        years = self.balance_sheet.columns

        # Score 1: - Return on Assets
        self.net_income = self.income_statement[years[0]]['NetIncome'] / 1000
        self.total_assets = self.balance_sheet[years[0]]['TotalAssets']
        self.avg_total_assets = (self.total_assets + self.balance_sheet[years[1]]['TotalAssets']) / 2
        self.rota = self.net_income / self.total_assets
        self.roaa = self.net_income / self.avg_total_assets
        roaa_score = 1 if self.roaa > 0 else 0
            
        # Score 2: Operating Cash Flow
        self.operating_cashflow = self.cashflow_statement[years[0]]['CashFlowFromContinuingOperatingActivities']
        operating_cashflow_score = 1 if self.operating_cashflow > 0 else 0

        # Score 3: Change in ROA
        avg_total_assets_py = (self.balance_sheet[years[1]]['TotalAssets'] + self.balance_sheet[years[2]]['TotalAssets']) / 2
        net_income_py = self.income_statement[years[1]]['NetIncome'] / 1000
        self.roa = self.net_income / self.avg_total_assets
        self.roa_py = net_income_py / avg_total_assets_py
        roa_score = 1 if self.roa > 0 else 0

        # Score 4: Accruals
        self.accruals = self.operating_cashflow / self.total_assets - self.roa
        ac_score = 1 if self.accruals > 0 else 0

        self.profitability_score = roaa_score + operating_cashflow_score + roa_score + ac_score

        profitability = {
            'Net Income': self.net_income,
            'ROAA': self.roaa,
            'ROAA Score': roaa_score,
            'Operating Cash Flow': self.operating_cashflow,
            'Operating Cash Flow Score': operating_cashflow_score,
            'ROA': self.roa,
            'ROA PY': self.roa_py,
            'ROA Score': roa_score,
            'Accruals': self.accruals,
            'Accruals Score': ac_score,
            'Profitability Score': self.profitability_score
        }

        return profitability

    def get_leverage(self) -> dict:
        years = self.balance_sheet.columns

        # Score 5: Change in Leverage
        try:
            self.long_term_debt = self.balance_sheet[years[0]]['LongTermDebt']
            self.total_debt = self.balance_sheet[years[0]]['TotalDebt']
            self.long_term_debt_ratio = self.long_term_debt / self.total_debt
            long_term_debt_ratio_py = self.balance_sheet[years[1]]['LongTermDebt'] / self.balance_sheet[years[1]]['TotalDebt']
            long_term_debt_score = 1 if self.long_term_debt_ratio < long_term_debt_ratio_py else 0
        except:
            long_term_debt_score = 0

        # Score 6: Change in Current Ratio
        try:
            self.current_assets = self.balance_sheet[years[0]]['CurrentAssets']
            self.current_liabilities = self.balance_sheet[years[0]]['CurrentLiabilities']
            self.current_ratio = self.current_assets / self.current_liabilities
            current_ratio_py = self.balance_sheet[years[1]]['CurrentAssets'] / self.balance_sheet[years[1]]['CurrentLiabilities']
            current_ratio_score = 1 if self.current_ratio > current_ratio_py else 0
        except:
            current_ratio_score = 0

        # Score 7: Change in number of shares
        try:
            self.shares_outstanding = self.balance_sheet[years[0]]['CommonStock']
            shares_outstanding_py = self.balance_sheet[years[1]]['CommonStock']
            shares_outstanding_score = 1 if self.shares_outstanding < shares_outstanding_py else 0
        except:
            shares_outstanding_score = 0

        self.leverage_score = long_term_debt_score + current_ratio_score + shares_outstanding_score
        try:
            leverage = {
                'Long Term Debt Ratio': self.long_term_debt_ratio,
                'Long Term Debt Score': long_term_debt_score,
                'Current Ratio': self.current_ratio,
                'Current Ratio Score': current_ratio_score,
                'Shares Outstanding': self.shares_outstanding,
                'Shares Outstanding Score': shares_outstanding_score,
                'Leverage Score': self.leverage_score
            }
        except:
            leverage = {
                'Long Term Debt Ratio': 0,
                'Long Term Debt Score': 0,
                'Leverage Score': 0,
            }

        return leverage

    def get_operating_efficiency(self) -> dict:
        years = self.balance_sheet.columns

        # Score 8: Change in Gross Margin
        try:
            self.gross_profit = self.income_statement[years[0]]['GrossProfit']
            self.revenue = self.income_statement[years[0]]['TotalRevenue']
            self.gross_margin = self.gross_profit / self.revenue
            gross_margin_py = self.income_statement[years[1]]['GrossProfit'] / self.income_statement[years[1]]['TotalRevenue']
            gross_margin_score = 1 if self.gross_margin > gross_margin_py else 0
        except:
            gross_margin_score = 0

        # Score 9: Change in Asset Turnover
        try:
            self.asset_turnover = self.revenue / self.avg_total_assets
            asset_turnover_py = self.income_statement[years[1]]['TotalRevenue'] / (self.balance_sheet[years[1]]['TotalAssets'] + self.balance_sheet[years[2]]['TotalAssets']) / 2
            asset_turnover_score = 1 if self.asset_turnover > asset_turnover_py else 0
        except:
            asset_turnover_score = 0

        # Score 10: Change in Inventory Turnover
        try:
            self.inventory = self.balance_sheet[years[0]]['Inventory']
            self.cogs = self.income_statement[years[0]]['CostOfRevenue']
            self.inventory_turnover = self.cogs / self.inventory
            inventory_turnover_py = self.income_statement[years[1]]['CostOfRevenue'] / self.balance_sheet[years[1]]['Inventory']
            self.inventory_turnover_score = 1 if self.inventory_turnover > inventory_turnover_py else 0

            self.operating_efficiency_score = gross_margin_score + asset_turnover_score 

            operating_efficiency = {
                'Gross Margin': self.gross_margin,
                'Gross Margin Score': gross_margin_score,
                'Asset Turnover': self.asset_turnover,
                'Asset Turnover Score': asset_turnover_score,
                'Inventory Turnover': self.inventory_turnover,
                'Inventory Turnover Score': self.inventory_turnover_score,
                'Operating Efficiency Score': self.operating_efficiency_score
            }
            return operating_efficiency


        except:

            self.operating_efficiency_score = 0

            operating_efficiency = {
                'Gross Margin': 0,
                'Gross Margin Score': 0,
                'Asset Turnover': 0,
                'Asset Turnover Score': 0,
                'Inventory Turnover': 0,
                'Inventory Turnover Score': 0,
                'Operating Efficiency Score': self.operating_efficiency_score
            }
            return operating_efficiency
    
    def get_financials(self) -> dict:
        self.financials = self.Ticker.financials
        return self.financials
            
    def get_ratios(self) -> dict:
        self.ratios = {}
        try:
            self.ratios['current_ratio'] = self.current_ratio
        except:
            self.ratios['current_ratio'] = 0
        self.ratios['price_earnings_ratio'] = self.info['trailingPE']
        self.ratios['projected_price_earnings_ratio'] = self.info['forwardPE']
        self.ratios['profit_sales_ratio'] = self.info['priceToSalesTrailing12Months']
        self.ratios['profit_margin'] = self.info['profitMargins']
        self.ratios['profit_book_ratio'] = self.info['priceToBook']
        # self.ratios['profit_cash_ratio'] = self.info['priceToCashflow']
        try:
            self.ratios['profit_dividend_ratio'] = self.info['dividendYield']
        except:
            self.ratios['profit_dividend_ratio'] = 0
        self.ratios['profit_earnings_growth'] = self.info['pegRatio']
        self.ratios['profit_sales_growth'] = self.info['revenueGrowth']
        return self.ratios

    def get_piotroski(self) -> dict:
        self.f_score = self.profitability['Profitability Score'] + self.leverage['Leverage Score'] + self.operating_efficiency['Operating Efficiency Score']
        
        piotroski = {
            'Profitability Score': self.profitability['Profitability Score'],
            'Leverage Score': self.leverage['Leverage Score'],
            'Operating Efficiency Score': self.operating_efficiency['Operating Efficiency Score'],
            'F-Score': self.f_score
        }
        return piotroski

    def get_altman_z_score(self):
        try:
            altman_z_score = 1.2 * self.profitability['ROAA'] + 1.4 * self.leverage['Current Ratio'] + 3.3 * self.operating_efficiency['Asset Turnover'] + 0.6 * self.leverage['Long Term Debt Ratio'] + 1.0 * self.profitability['Accruals']
            return altman_z_score
        except:
            return None
    
    def plot(self):
        plt.figure(figsize=(15, 7))
        plt.plot(self.df['Adj Close'], label=(self.symbol + " Adj Close"), color='blue')
        plt.title(self.symbol)
        plt.legend()
        plt.show()


class TechnicalAnalysis:
    # def __init__(self, ticker: Ticker):
    def __init__(self, ticker: str, ticker_df: pd.DataFrame):
        # self.Ticker = ticker
        # self.start = self.Ticker.start
        # self.end = self.Ticker.end
        # self.symbol = self.Ticker.symbol
        self.symbol = ticker
        # self.df = self.Ticker.df.copy(deep=True)
        self.df = ticker_df.copy(deep=True)
        self.df_ta = self.df.copy(deep=True)
        # self.add_basic_indicators()
        # self.add_bollinger_bands()
        self.add_z_score()
        # self.short_term_moving_averages()
        # self.long_term_moving_averages()
        # self.short_exponential_moving_averages()
        # self.long_exponential_moving_averages()
        self.add_macd()
        self.add_momentum_indicators()
        # self.add_volume_indicators()

        # for method in dir(self):
        #     if method.startswith('add_'):
        #         df_name = method.split('add_')[1] + '_df'
        #         self.df_name = self.df.copy(deep=True)
        #         self.df_name = getattr(self, method)()
    
    def __str__(self):
        return self.symbol
    
    def __repr__(self):
        return self.symbol
    
    def add_basic_indicators(self) -> pd.DataFrame:
        self.df_ta['ma_20'] = self.df_ta['Adj Close'].rolling(window=20).mean()
        self.df_ta['vol'] = self.df_ta['Adj Close'].rolling(window=20).std()
        self.df_ta['upper_bb'] = self.df_ta['ma_20'] + (self.df_ta['vol'] * 2)
        self.df_ta['lower_bb'] = self.df_ta['ma_20'] - (self.df_ta['vol'] * 2)
        self.df_ta['rsi'] = ta.momentum.rsi(self.df_ta['Adj Close'], window=6)

    def add_bollinger_bands(self) -> pd.DataFrame:
        self.df_ta['BollingerBandHigh'] = self.df_ta['ma_20'] + (self.df_ta['vol'] * 2)
        self.df_ta['BollingerBandLow'] = self.df_ta['ma_20'] - (self.df_ta['vol'] * 2)
        self.df_ta['BollingerMAvg'] = ta.volatility.bollinger_mavg(self.df_ta['Adj Close'], window=20)
        self.df_ta['Bollinger%b'] = ta.volatility.bollinger_pband(self.df_ta['Adj Close'])
        self.df_ta['Bollinger Width'] = ta.volatility.bollinger_wband(self.df['Adj Close'])

    def add_z_score(self, frame: str = 'Adj Close', window1 = 20) -> pd.DataFrame:
        self.df_ta['Z Score ' + frame] = (self.df[frame] - self.df[frame].rolling(window= window1).mean()) / self.df[frame].rolling(window= window1).std()
        return self.df_ta

    def short_term_moving_averages(self) -> pd.DataFrame:
        self.df_short_sma = self.df.copy(deep=True)
        self.df_short_sma['SMA_10'] = ta.trend.sma_indicator(self.df['Adj Close'], window=10)
        self.df_short_sma['SMA_20'] = ta.trend.sma_indicator(self.df['Adj Close'], window=20)
        self.df_short_sma['SMA_30'] = ta.trend.sma_indicator(self.df['Adj Close'], window=30)
        return self.df_short_sma

    def long_term_moving_averages(self) -> pd.DataFrame:
        self.df_long_sma = self.df.copy(deep=True)
        self.df_long_sma['SMA_50'] = ta.trend.sma_indicator(self.df['Adj Close'], window=50)
        self.df_long_sma['SMA_100'] = ta.trend.sma_indicator(self.df['Adj Close'], window=100)
        self.df_long_sma['SMA_200'] = ta.trend.sma_indicator(self.df['Adj Close'], window=200)
        return self.df_long_sma

    def short_exponential_moving_averages(self) -> pd.DataFrame:
        self.df_short_exponential = self.df.copy(deep=True)
        self.df_short_exponential['EMA_10'] = ta.trend.ema_indicator(self.df['Adj Close'], window=10)
        self.df_short_exponential['EMA_20'] = ta.trend.ema_indicator(self.df['Adj Close'], window=20)
        self.df_short_exponential['EMA_30'] = ta.trend.ema_indicator(self.df['Adj Close'], window=30)
        return self.df_short_exponential

    def long_exponential_moving_averages(self) -> pd.DataFrame:
        self.df_long_exponential = self.df.copy(deep=True)
        self.df_long_exponential['EMA_50'] = ta.trend.ema_indicator(self.df['Adj Close'], window=50)
        self.df_long_exponential['EMA_100'] = ta.trend.ema_indicator(self.df['Adj Close'], window=100)
        self.df_long_exponential['EMA_200'] = ta.trend.ema_indicator(self.df['Adj Close'], window=200)
        return self.df_long_exponential

    def add_macd(self) -> pd.DataFrame:
        self.df_ta['MACD'] = ta.trend.macd_diff(self.df['Adj Close'])
        self.df_ta['MACD Signal'] = ta.trend.macd_signal(self.df['Adj Close'])
        self.df_ta['MACD Histogram'] = ta.trend.macd_diff(self.df['Adj Close']) - ta.trend.macd_signal(self.df['Adj Close'])

    def add_keltner_channels(self) -> pd.DataFrame:
        self.df_ta['KeltnerH'] = ta.volatility.keltner_channel_hband(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['KeltnerL'] = ta.volatility.keltner_channel_lband(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['KeltnerWidth'] = ta.volatility.keltner_channel_wband(self.df['High'], self.df['Low'], self.df['Adj Close'])

    def add_donchian_channels(self) -> pd.DataFrame:
        self.df_ta['DonchianH'] = ta.volatility.donchian_channel_hband(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['DonchianL'] = ta.volatility.donchian_channel_lband(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['DonchianWidth'] = ta.volatility.donchian_channel_wband(self.df['High'], self.df['Low'], self.df['Adj Close'])

    def add_vortex_indicator(self) -> pd.DataFrame:
        self.df_ta['Vortex+'] = ta.trend.vortex_indicator_pos(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['Vortex-'] = ta.trend.vortex_indicator_neg(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['VortexSignal'] = self.df['Vortex+'] - self.df['Vortex-']

    def add_ichimoku_cloud(self) -> pd.DataFrame:
        self.df_ta['IchimokuA'] = ta.trend.ichimoku_a(self.df['High'], self.df['Low'])
        self.df_ta['IchimokuB'] = ta.trend.ichimoku_b(self.df['High'], self.df['Low'])
        self.df_ta['IchimokuBaseLine'] = ta.trend.ichimoku_base_line(self.df['High'], self.df['Low'])
        self.df_ta['IchimokuConvLine'] = ta.trend.ichimoku_conversion_line(self.df['High'], self.df['Low'])

    def add_aroon(self) -> pd.DataFrame:
        self.df_ta['AroonUp'] = ta.trend.aroon_up(self.df['High'], self.df['Low'])
        self.df_ta['AroonDown'] = ta.trend.aroon_down(self.df['High'], self.df['Low'])
        self.df_ta['AroonInd'] = self.df['AroonUp'] - self.df['AroonDown']

    def add_kst(self) -> pd.DataFrame:
        self.df_ta['KST'] = ta.trend.kst(self.df['Adj Close'])
        self.df_ta['KSTsig'] = ta.trend.kst_sig(self.df['Adj Close'])
        self.df_ta['KSTdiff'] = self.df['KST'] - self.df['KSTsig']

    def add_momentum_indicators(self) -> pd.DataFrame:
        self.df_ta['RSI'] = ta.momentum.rsi(self.df['Adj Close'], window=6)
        self.df_ta['TSI'] = ta.momentum.tsi(self.df['Adj Close'])
        self.df_ta['Awesome Oscillator'] = ta.momentum.awesome_oscillator(self.df['High'], self.df['Low'])
        self.df_ta['Ultimate Oscillator'] = ta.momentum.ultimate_oscillator(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['Stoch'] = ta.momentum.stoch(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['Stoch Signal'] = ta.momentum.stoch_signal(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['Williams %R'] = ta.momentum.williams_r(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['KAMA'] = ta.momentum.kama(self.df['Adj Close'])
        self.df_ta['PPO'] = ta.momentum.ppo(self.df['Adj Close'])
        self.df_ta['ROC'] = ta.momentum.roc(self.df['Adj Close'])

    def add_trend_indicators(self) -> pd.DataFrame:
        self.df_ta['ADX'] = ta.trend.adx(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['CCI'] = ta.trend.cci(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['DPO'] = ta.trend.dpo(self.df['Adj Close'])
        self.df_ta['EMA'] = ta.trend.ema_indicator(self.df['Adj Close'])
        self.df_ta['Ichimoku'] = ta.trend.ichimoku_a(self.df['High'], self.df['Low'])
        self.df_ta['KST'] = ta.trend.kst(self.df['Adj Close'])
        self.df_ta['MACD'] = ta.trend.macd_diff(self.df['Adj Close'])
        self.df_ta['Mass Index'] = ta.trend.mass_index(self.df['High'], self.df['Low'])
        self.df_ta['Parabolic SAR'] = ta.trend.psar_up(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['TRIX'] = ta.trend.trix(self.df['Adj Close'])
        self.df_ta['Stooq'] = ta.trend.stc(self.df['Adj Close'])

    def add_volume_indicators(self) -> pd.DataFrame:
        self.df_ta['VWAP'] = ta.volume.volume_weighted_average_price(self.df['High'], self.df['Low'], self.df['Adj Close'], self.df['Volume'])
        self.df_ta['ForceIndex'] = ta.volume.force_index(self.df['Adj Close'], self.df['Volume'])
        self.df_ta['Accumulation/Distribution'] = ta.volume.acc_dist_index(self.df['High'], self.df['Low'], self.df['Adj Close'], self.df['Volume'])
        self.df_ta['Chaikin'] = ta.volume.chaikin_money_flow(self.df['High'], self.df['Low'], self.df['Adj Close'], self.df['Volume'])
        self.df_ta['EOM'] = ta.volume.ease_of_movement(self.df['High'], self.df['Low'], self.df['Adj Close'], self.df['Volume'])
        self.df_ta['FI'] = ta.volume.force_index(self.df['Adj Close'], self.df['Volume'])
        self.df_ta['MFI'] = ta.volume.money_flow_index(self.df['High'], self.df['Low'], self.df['Adj Close'], self.df['Volume'])
        self.df_ta['OBV'] = ta.volume.on_balance_volume(self.df['Adj Close'], self.df['Volume'])
        self.df_ta['VPT'] = ta.volume.volume_price_trend(self.df['Adj Close'], self.df['Volume'])

    def add_volatility_indicators(self) -> pd.DataFrame:
        self.df_ta['ATR'] = ta.volatility.average_true_range(self.df['High'], self.df['Low'], self.df['Adj Close'])
        self.df_ta['Ulcer Index'] = ta.volatility.ulcer_index(self.df['Adj Close'])

    def plot_z_score(self):
        plt.figure(figsize=(20, 10))
        self.z_score_df = self.df[['Adj Close', 'Z Score Adj Close']].copy(deep=True)
        self.z_score_df.dropna(inplace=True)
        plt.plot(self.z_score_df['Adj Close'], label=self.symbol, alpha=1, color='black')
        plt.plot(self.z_score_df['Z Score Adj Close'], label='Z Score', alpha=0.5)
        plt.legend()
        plt.show()
        plt.close()

    def plot_short_sma(self):
        plt.figure(figsize=(20, 10))
        self.short_sma_df = self.df[['Adj Close', 'SMA_10', 'SMA_20', 'SMA_30']].copy(deep=True)
        self.short_sma_df.dropna(inplace=True)
        plt.plot(self.short_sma_df['Adj Close'], label=self.symbol, alpha=1, color='black')
        plt.plot(self.short_sma_df['SMA_10'], label='SMA 10', alpha=0.5)
        plt.plot(self.short_sma_df['SMA_20'], label='SMA 20', alpha=0.5)
        plt.plot(self.short_sma_df['SMA_30'], label='SMA 30', alpha=0.5)
        plt.legend()
        plt.show()
        plt.close()

    def plot_long_sma(self):
        plt.figure(figsize=(20, 10))
        self.long_sma_df = self.df[['Adj Close', 'SMA_50', 'SMA_100', 'SMA_200']].copy(deep=True)
        self.long_sma_df.dropna(inplace=True)
        plt.plot(self.long_sma_df['Adj Close'], label=self.symbol, alpha=1, color='black')
        plt.plot(self.long_sma_df['SMA_50'], label='SMA 50', alpha=0.5)
        plt.plot(self.long_sma_df['SMA_100'], label='SMA 100', alpha=0.5)
        plt.plot(self.long_sma_df['SMA_200'], label='SMA 200', alpha=0.5)
        plt.legend()
        plt.show()
        plt.close()


    def plot_macd(self):
        plt.figure(figsize=(20, 10))
        self.macd_df = self.df[['MACD', 'MACD Signal', 'MACD Histogram']].copy(deep=True)
        self.macd_df.dropna(inplace=True)
        plt.plot(self.macd_df['MACD'], label='MACD', alpha=1, color='black')
        plt.plot(self.macd_df['MACD Signal'], label='MACD Signal', alpha=0.5)
        plt.bar(self.macd_df.index, self.macd_df['MACD Histogram'], label='MACD Histogram', alpha=0.5)
        plt.legend()
        plt.show()
        plt.close()


    def plot_bollinger_bands(self):
        plt.figure(figsize=(20, 10))
        self.df_ta.dropna(inplace=True)
        plt.plot(self.df_ta['Adj Close'], label=self.symbol, alpha=1)
        plt.plot(self.df_ta['ma_20'], label='Moving Avg(20 day)', alpha=0.70)
        plt.plot(self.df_ta['upper_bb'], label='Bollinger Band High', alpha=0.55)
        plt.plot(self.df_ta['lower_bb'], label='Bollinger Band Low', alpha=0.55)
        plt.legend()
        plt.show()
        plt.close()

    def generate_correlation_matrix(self):
        self.df_ta.dropna(inplace=True)
        self.df_ta.drop(columns=['Date'], inplace=True)
        self.correlation_matrix = self.df_ta.corr()
        return self.correlation_matrix

    def autocorrelation_plot(self):
        self.autocorrelation_df = self.df_ta[['Adj Close']].copy(deep=True)
        self.autocorrelation_df.dropna(inplace=True)
        autocorrelation_plot(self.autocorrelation_df)
        plt.show()
        plt.close()

    # def generate_prophet_forecast(self, frame= 'Adj Close' ,period=90):
    #     self.prophet_df = self.df[[frame]].copy(deep=True)
    #     self.prophet_df = self.prophet_df.reset_index()
    #     self.prophet_df = self.prophet_df.rename(columns={'Date': 'ds', frame: 'y'})
    #     self.model = Prophet()
    #     self.model.fit(self.prophet_df)
    #     self.future = self.model.make_future_dataframe(periods= period)
    #     self.forecast = self.model.predict(self.future)
    #     self.model.plot(self.forecast)
    #     self.model.plot_components(self.forecast)
    #     plt.show()
    #     plt.close()

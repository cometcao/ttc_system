# -*- encoding: utf8 -*-
'''
Created on 2 Aug 2017

@author: MetalInvest
'''

try:
    from rqdatac import *
except:
    pass 
import numpy as np
import pandas as pd
import math
from sklearn.svm import SVR  
from sklearn.model_selection import GridSearchCV  
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class ML_Factor_Rank(object):
    '''
    This class use regression to periodically gauge stocks fartherest away from the 
    regressed frontier, and return the list of stocks
    '''
    def __init__(self, params):
        self.stock_num = params.get('stock_num', 5)
        self.index_scope = params.get('index_scope', '000985.XSHG')
    
    def gaugeStocks(self, context):
        pre_today = get_previous_trading_date(context.now.date())
        sample = index_components(self.index_scope, date = pre_today)
        q = query(fundamentals.stockcode, 
                  fundamentals.eod_derivative_indicator.market_cap, 
                  fundamentals.balance_sheet.total_assets - fundamentals.balance_sheet.total_liability,
                  fundamentals.balance_sheet.total_assets / fundamentals.balance_sheet.total_liability, 
                  fundamentals.income_statement.net_profit, 
                  fundamentals.income_statement.net_profit + 1, 
                  fundamentals.financial_indicator.inc_revenue).filter(fundamentals.stockcode.in_(sample))
        df = get_fundamentals(q, date = pre_today)
        df.columns = ['code', 'log_mcap', 'log_NC', 'LEV', 'NI_p', 'NI_n', 'g']
        
        df['log_mcap'] = np.log(df['log_mcap'])
        df['log_NC'] = np.log(df['log_NC'])
        df['NI_p'] = np.log(np.abs(df['NI_p']))
        df['NI_n'] = np.log(np.abs(df['NI_n'][df['NI_n']<0]))
        df.index = df.code.values
        del df['code']
        df = df.fillna(0)
        df[df>10000] = 10000
        df[df<-10000] = -10000
        industry_set = ['801010', '801020', '801030', '801040', '801050', '801080', '801110', '801120', '801130', 
                  '801140', '801150', '801160', '801170', '801180', '801200', '801210', '801230', '801710',
                  '801720', '801730', '801740', '801750', '801760', '801770', '801780', '801790', '801880','801890']
        
        for i in range(len(industry_set)):
            industry = get_industry_stocks(industry_set[i], date = pre_today)
            s = pd.Series([0]*len(df), index=df.index)
            s[set(industry) & set(df.index)]=1
            df[industry_set[i]] = s
            
        X = df[['log_NC', 'LEV', 'NI_p', 'NI_n', 'g', 'log_RD','801010', '801020', '801030', '801040', '801050', 
                '801080', '801110', '801120', '801130', '801140', '801150', '801160', '801170', '801180', '801200', 
                '801210', '801230', '801710', '801720', '801730', '801740', '801750', '801760', '801770', '801780', 
                '801790', '801880', '801890']]
        Y = df[['log_mcap']]
        X = X.fillna(0)
        Y = Y.fillna(0)
        
        svr = SVR(kernel='rbf', gamma=0.1) 
        model = svr.fit(X, Y)
        factor = Y - pd.DataFrame(svr.predict(X), index = Y.index, columns = ['log_mcap'])
        factor = factor.sort_index(by = 'log_mcap')
        stockset = list(factor.index[:10])
# encoding: utf-8
'''
Created on 26 Sep 2017

@author: MetalInvest
'''
try:
    from rqdatac import *
except:
    pass   
import pandas as pd
import numpy as np

class HerdHead(object):

    def __init__(self, params):
        self.gainThre = params.get('gainThre', 0.05)
        self.count = params.get('count', 20)
        self.period = params.get('period', '1d')
        self.useIntraday = params.get('useIntraday', True)
        self.intraday_period = params.get('intraday_period', '230m')
        self.isAnal = params.get('isAnal', False)
    
    def filterStocks(self, stock_list):
        current_data = get_current_data()
        stock_list = [stock for stock in stock_list
                      if not current_data[stock].is_st
                      and 'ST' not in current_data[stock].name
                      and '*' not in current_data[stock].name
                      and '退' not in current_data[stock].name]
        stock_list = [stock for stock in stock_list if not current_data[stock].paused]
        return stock_list
    
    # 寻找指数的龙头股
    def findLeadStock(self,index,method = 0,isConcept=False):
        # 规则
        # 1.涨幅大于阈值的股票；
        # 2.指数涨幅在阈值的四分之一以上；
        # 3.过去一周成交量大于过去两周成交量；
        # 4.个股流通市值占总市值百分比达到阈值
        
        # 取出该指数的股票:
        oriStocks = get_industry_stocks(index) if not isConcept else get_concept_stocks(index)
        if not self.isAnal:
            oriStocks = self.filterStocks(oriStocks)
        # 根据个股涨幅筛选
        filtStocks = self.filtGain(oriStocks)
        # 计算指数涨幅
        gainIndex = self.calIndustryGain(oriStocks)
        
        # 根据规则筛选龙头股
        if float(gainIndex)/self.gainThre > 0.25:
            if method == 0:
                # 基本限制
                return filtStocks
            elif method == 1:
                # 基本限制+成交量限制
                filtStocks = self.filtVol(filtStocks)
                return filtStocks
            elif method == 2:
                # 基本限制+流通市值限制
                filtStocks = self.filtMarketCap(filtStocks, oriStocks)
                return filtStocks
            elif method == 3:
                # 基本限制+流通市值限制+成交量限制
                filtStocks = self.filtVol(filtStocks)
                if len(filtStocks) != 0:
                    filtStocks = self.filtMarketCap(filtStocks,oriStocks)
                else:
                    pass
                return filtStocks
            else:
                return 'Error method order'
        else:
            return []
        
    # 根据涨幅筛选股票
    def filtGain(self, stocks):
        # 初始化参数信息
        rankValue = []
    
        # 计算涨跌幅
        for security in stocks:
            # 获取过去pastDay的指数值
            stocksPrice = self.getlatest_df(security, self.count, fields=['close'], skip_paused=True, df_flag=True)
            if len(stocksPrice)!=0:
                # 计算涨跌幅
                errCloseOpen = [(float(stocksPrice.iloc[-1]) - float(stocksPrice.iloc[0])) / float(stocksPrice.iloc[0])]
                rankValue += errCloseOpen
            else:
                rankValue += [0]
    
        # 根据周涨跌幅排名
        filtStocks = {'code':stocks,'rankValue':rankValue}
        filtStocks = pd.DataFrame(filtStocks)
        filtStocks = filtStocks.sort('rankValue',ascending = False)
        filtStocks = filtStocks[filtStocks['rankValue'] > self.gainThre]
        filtStocks = list(filtStocks['code'])
        return filtStocks
    
    # 根据成交量筛选股票
    def filtVol(self, stocks):
        # 初始化返回数组
        returnStocks = []
        # 筛选
        stockVol = history(self.count, unit=self.period, field='volume', security_list=stocks, df=False, skip_paused=False, fq='pre')
        if self.useIntraday:
            stockVol_intraday = history(1, unit=self.intraday_period, field='volume', security_list=stocks, df=False, skip_paused=False, fq='pre')
            for security in stocks:
                stockVol[security] = np.append(stockVol[security], stockVol_intraday[security])
        
        for security in stocks:
            if float(stockVol[security][-5:].mean()) > float(stockVol[security][-10:].mean()):
                returnStocks += [security]
            else:
                continue
        return returnStocks
    # 根据流通市值筛选股票
    def filtMarketCap(self,stocks,oriStocks):
        # 初始化返回数组
        returnStocks = []
    
        # 计算行业流通市值
        indexMarketCap = get_fundamentals(query(valuation.circulating_market_cap).filter(valuation.code.in_(oriStocks)))
        totalMarketCap = float(sum(indexMarketCap['circulating_market_cap']))
        
        # 计算个股流通市值占总市值百分比阈值：以四分位为阈值
        indexMarketCap = indexMarketCap.div(totalMarketCap,axis=0)
        porThre = indexMarketCap.describe()
        porThre = float(porThre.loc['25%'])
    
        # 筛选
        for security in stocks:
            stockMarketCap = get_fundamentals(query(valuation.circulating_market_cap).filter(valuation.code.in_([security])))
            if float(stockMarketCap.iloc[0]) > totalMarketCap * porThre:
                returnStocks += [security]
            else:
                continue
        return returnStocks
    
    # 计算行业涨幅
    def calIndustryGain(self, stocks):
        gainIndex = 0
        if not stocks:
            return 0
        for security in stocks:
            stocksPrice = self.getlatest_df(security, self.count, fields=['close'], skip_paused=True, df_flag=True)
            if len(stocksPrice) != 0:
                gainIndex += (float(stocksPrice.iloc[-1]) - float(stocksPrice.iloc[0])) / float(stocksPrice.iloc[0])
            else:
                continue
        return gainIndex/len(stocks)
    
    def getlatest_df(self, stock, count, fields, skip_paused=True, df_flag = True):
        df = attribute_history(stock, count, unit=self.period, fields = fields, skip_paused=skip_paused, df=df_flag)
        if self.useIntraday:
            containPaused = 'paused' in fields
            if containPaused:
                fields.remove('paused')
            latest_stock_data = attribute_history(stock, 1, self.intraday_period, fields, skip_paused=skip_paused, df=df_flag)
            if containPaused:
                latest_stock_data.assign(paused=np.nan)
                cd = get_current_data()
                latest_stock_data.ix[-1,'paused'] = cd[stock].paused

            if df_flag:
                current_date = latest_stock_data.index[-1].date()
                latest_stock_data = latest_stock_data.reset_index(drop=False)
                latest_stock_data.ix[0, 'index'] = pd.DatetimeIndex([current_date])[0]
                latest_stock_data = latest_stock_data.set_index('index')
                df = df.reset_index().drop_duplicates(subset='index').set_index('index')
                df = df.append(latest_stock_data, verify_integrity=True) # True
            else:
                final_fields = []
                if isinstance(fields, basestring):
                    final_fields.append(fields)
                else:
                    final_fields = list(fields)
#                 [np.append(df[field], latest_stock_data[field][-1]) for field in final_fields]
                for field in final_fields:
                    df[field] = np.append(df[field], latest_stock_data[field][-1])
        return df
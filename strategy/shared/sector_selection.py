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
import talib
import datetime
from sector_spider import *
from securityDataManager import *

def get_data(stock, count, level, fields, skip_paused=False, df_flag=True, isAnal=False, includenow=False):
    df = None
    if isAnal:
        latest_trading_day = datetime.datetime.now().date()
        start_date = get_trading_dates('2006-01-01', latest_trading_day)[-self.count]
        df = SecurityDataManager.get_research_data_rq(stock, start_date=start_date, end_date=latest_trading_day, period=level, fields = fields, skip_suspended=skip_paused, df=df_flag)
    else:
        df = SecurityDataManager.get_data_rq(stock, count, level, fields=fields, skip_suspended=skip_paused, df=df_flag, include_now=includenow)
    return df

class SectorSelection(object):
    '''
    This class implement the methods to rank the sectors
    '''
    def __init__(self, isAnal=False, limit_pct=5, isStrong=True, min_max_strength = 0, useIntradayData=True, useAvg=True, avgPeriod=5, intraday_period='230m', context=None):
        '''
        Constructor
        '''
        self.useIntradayData = useIntradayData
        self.useAvg = useAvg
        self.isAnal = isAnal
        self.frequency = '1d' # use day period
        self.period = 270
        self.gauge_period = avgPeriod
        self.top_limit = float(limit_pct) / 100.0
        self.isReverse = isStrong
        self.stock_data_buffer = {}
        self.min_max_strength = min_max_strength
        self.intraday_period = intraday_period
        self.context=context
        
        ss = sectorSpider()
        self.jqIndustry = ss.getSectorCode('zjh') 
        self.conceptSectors = []
        self.filtered_industry = []
        self.filtered_concept = []

    def displayResult(self, industryStrength, isConcept=False):
        limit_value = int(self.top_limit * len(self.conceptSectors) if isConcept else self.top_limit * len(self.jqIndustry))
        for sector, strength in industryStrength[:limit_value]:
            stocks = []
            if isConcept:
                stocks = concept(sector)
            else:
                stocks = industry(sector)
            print (sector+'@'+str(strength)+':'+','.join([instruments(s).symbol for s in stocks]))
            
    def sendResult(self, industryStrength, isConcept=False):
        message = ""
        limit_value = int(self.top_limit * len(self.conceptSectors) if isConcept else self.top_limit * len(self.jqIndustry))
        for sector, strength in industryStrength[:limit_value]:
            stocks = []
            if isConcept:
                stocks = concept(sector)
            else:
                stocks = industry(sector)
            message += sector + ':'
            message += ','.join([instruments(s).symbol for s in stocks])
            message += '***'
        send_message(message, channel='weixin')      

    def processAllSectors(self, sendMsg=False, display=False):
        if self.filtered_industry: # ignore concept case here
            print ("use cached sectors")
            return (self.filtered_industry, self.filtered_concept)
        
        industryStrength = self.processIndustrySectors()
        conceptStrength = self.processConceptSectors()
        if display:
            self.displayResult(industryStrength)
            self.displayResult(conceptStrength, True)
        if sendMsg:
            self.sendResult(industryStrength)
            self.sendResult(conceptStrength, True)
        concept_limit_value = int(self.top_limit * len(self.conceptSectors))
        industry_limit_value = int(self.top_limit * len(self.jqIndustry))
        self.filtered_industry = [sector for sector, strength in industryStrength[:industry_limit_value] if (strength >= self.min_max_strength if self.isReverse else strength <= self.min_max_strength)] 
        self.filtered_concept = [sector for sector, strength in conceptStrength[:concept_limit_value] if (strength >= self.min_max_strength if self.isReverse else strength <= self.min_max_strength)]
        return (self.filtered_industry, self.filtered_concept)
    
    def processAllSectorStocks(self, isDisplay=False):
        all_industry, all_concept = self.processAllSectors(display=isDisplay)
        allstocks = []
        for idu in all_industry:
            allstocks += industry(idu)
        for con in all_concept:
            allstocks += concept(con)
        return list(set(allstocks))
        
    def processIndustrySectors(self):
        industryStrength = []

        for indu in self.jqIndustry:
            try:
                stocks = industry(indu)
            except Exception as e:
                print(str(e))
                continue
            if len(stocks) > 3:
                industryStrength.append((indu, self.gaugeSectorStrength(stocks)))
        industryStrength = sorted(industryStrength, key=lambda x: x[1], reverse=self.isReverse)
        return industryStrength
    
    def processConceptSectors(self):
        # concept
        conceptStrength = []

        for con in self.conceptSectors:
            try:
                stocks = concept(con)
            except Exception as e:
                print(str(e))
                continue
            if len(stocks) > 3:
                conceptStrength.append((con, self.gaugeSectorStrength(stocks)))
            
        conceptStrength = sorted(conceptStrength, key=lambda x: x[1], reverse=self.isReverse)
        return conceptStrength
        
    def gaugeSectorStrength(self, sectorStocks):
        if not self.useAvg:
            sectorStrength = 0.0
            removed = 0
            for stock in sectorStocks:
                stockStrength = self.gaugeStockUpTrendStrength_MA(stock, isWeighted=False, index=-1)
                if stockStrength == -1:
                    removed+=1
                else:
                    sectorStrength += stockStrength
            if len(sectorStocks)==removed:
                sectorStrength = 0.0
            else:
                sectorStrength /= (len(sectorStocks)-removed)
            return sectorStrength  
        else:
            avgStrength = 0.0
            for i in range(-1, -self.gauge_period-1, -1): #range
                sectorStrength = 0.0
                removed = 0
                for stock in sectorStocks:
                    stockStrength = self.gaugeStockUpTrendStrength_MA(stock, isWeighted=False, index=i)
                    if stockStrength == -1:
                        removed+=1
                    else:
                        sectorStrength += stockStrength
                if len(sectorStocks)==removed:
                    sectorStrength = 0.0
                else:
                    sectorStrength /= (len(sectorStocks)-removed)
                avgStrength += sectorStrength
            avgStrength /= self.gauge_period
            return avgStrength    
    
    def gaugeStockUpTrendStrength_MA(self, stock, isWeighted=True, index=-1):
        today = self.context.now.date()
        stock_suspend_check = is_suspended(stock, end_date=today)
        if index == -1:
            stock_df = self.getlatest_df(stock, self.period, ['close'], skip_paused=True, df_flag=False)
            stock_df = np.array([data[0] for data in stock_df]) # hack the data remove tuple
            MA_5 = self.simple_moving_avg(stock_df, 5)
            MA_13 = self.simple_moving_avg(stock_df, 13)
            MA_21 = self.simple_moving_avg(stock_df, 21)
            MA_34 = self.simple_moving_avg(stock_df, 34)
            MA_55 = self.simple_moving_avg(stock_df, 55)
            MA_89 = self.simple_moving_avg(stock_df, 89)
            MA_144 = self.simple_moving_avg(stock_df, 144)
            MA_233 = self.simple_moving_avg(stock_df, 233)
            if (stock_suspend_check is not None and not stock_suspend_check.empty and stock_suspend_check.iloc[-1,0]) or stock_df.size==0: # paused we need to remove it from calculation
                return -1 
            elif stock_df[index] < MA_5 or np.isnan(MA_5):
                return 0 if isWeighted else 1
            elif stock_df[index] < MA_13 or np.isnan(MA_13):
                return 5 if isWeighted else 2
            elif stock_df[index] < MA_21 or np.isnan(MA_21):
                return 13 if isWeighted else 3
            elif stock_df[index] < MA_34 or np.isnan(MA_34):
                return 21 if isWeighted else 4
            elif stock_df[index] < MA_55 or np.isnan(MA_55):
                return 34 if isWeighted else 5
            elif stock_df[index] < MA_89 or np.isnan(MA_89):
                return 55 if isWeighted else 6
            elif stock_df[index] < MA_144 or np.isnan(MA_144):
                return 89 if isWeighted else 7
            elif stock_df[index] < MA_233 or np.isnan(MA_233):
                return 144 if isWeighted else 8
            else:
                return 233 if isWeighted else 9
        else: # take average value of past 20 periods
            stock_df = MA_5 = MA_13 = MA_21 = MA_34 = MA_55 = MA_89 = MA_144 = MA_233 = None
            try:
                if stock not in self.stock_data_buffer:
                    stock_df = self.getlatest_df(stock, self.period, ['close'], skip_paused=False, df_flag=True)
                    MA_5 = talib.SMA(stock_df.close.values, 5)
                    MA_13 = talib.SMA(stock_df.close.values, 13)
                    MA_21 = talib.SMA(stock_df.close.values, 21)
                    MA_34 = talib.SMA(stock_df.close.values, 34)
                    MA_55 = talib.SMA(stock_df.close.values, 55)
                    MA_89 = talib.SMA(stock_df.close.values, 89)
                    MA_144 = talib.SMA(stock_df.close.values, 144)
                    MA_233 = talib.SMA(stock_df.close.values, 233)
                    self.stock_data_buffer[stock]=[stock_df, MA_5, MA_13, MA_21, MA_34, MA_55, MA_89, MA_144, MA_233]
                else:
                    stock_df = self.stock_data_buffer[stock][0]
                    MA_5 = self.stock_data_buffer[stock][1]
                    MA_13 = self.stock_data_buffer[stock][2]
                    MA_21 = self.stock_data_buffer[stock][3]
                    MA_34 = self.stock_data_buffer[stock][4]
                    MA_55 = self.stock_data_buffer[stock][5]
                    MA_89 = self.stock_data_buffer[stock][6]
                    MA_144 = self.stock_data_buffer[stock][7]
                    MA_233 = self.stock_data_buffer[stock][8]
            except Exception as e:
                print (str(e))
                return -1
            if (not stock_suspend_check.empty and stock_suspend_check.iloc[-1,0]) or stock_df.empty: # paused we need to remove it from calculation
                return -1 
            elif stock_df.close[index] < MA_5[index] or np.isnan(MA_5[index]):
                return 0 if isWeighted else 1
            elif stock_df.close[index] < MA_13[index] or np.isnan(MA_13[index]):
                return 5 if isWeighted else 2
            elif stock_df.close[index] < MA_21[index] or np.isnan(MA_21[index]):
                return 13 if isWeighted else 3
            elif stock_df.close[index] < MA_34[index] or np.isnan(MA_34[index]):
                return 21 if isWeighted else 4
            elif stock_df.close[index] < MA_55[index] or np.isnan(MA_55[index]):
                return 34 if isWeighted else 5
            elif stock_df.close[index] < MA_89[index] or np.isnan(MA_89[index]):
                return 55 if isWeighted else 6
            elif stock_df.close[index] < MA_144[index] or np.isnan(MA_144[index]):
                return 89 if isWeighted else 7
            elif stock_df.close[index] < MA_233[index] or np.isnan(MA_233[index]):
                return 144 if isWeighted else 8
            else:
                return 233 if isWeighted else 9

    def simple_moving_avg(self, series, period):
        if len(series) < period:
            return 0
        total = sum(series[-period:])
        return total/period
    
    def getlatest_df(self, stock, count, fields, skip_paused=True, df_flag = True):
        df = get_data(stock, count, level='1d', fields=fields, skip_paused=skip_paused, df_flag=df_flag, isAnal=self.isAnal, includenow=self.useIntradayData)            
        return df
'''
Created on 15 Aug 2017

@author: MetalInvest
'''
try:
    from rqdatac import *
except:
    pass  
import types
import numpy as np
import pandas as pd
from biaoLiStatus import * 
from kBarProcessor import *
from securityDataManager import *

class ChanMatrix(object):
    '''
    classdocs
    '''
    gauge_level = ['5d', '1d', '60m']
    
    def __init__(self, stockList, isAnal=False):
        '''
        Constructor
        '''
        self.isAnal=isAnal
        self.count = 30 # 30
        self.stockList = stockList
        self.trendNodeMatrix = pd.DataFrame(index=self.stockList, columns=ChanMatrix.gauge_level)
    
    @classmethod
    def updateGauge_level(cls, value):
        cls.gauge_level = value
    
    def gaugeStockList(self, l=None):
        self.updateGaugeStockList(levels=ChanMatrix.gauge_level if not l else l)
        
    def updateGaugeStockList(self, levels, newStockList=None):
        candidate_list = newStockList if newStockList else self.stockList
        for stock in candidate_list:
            sc = self.gaugeStock_analysis(stock, levels) if self.isAnal else self.gaugeStock(stock, levels)
            for (level, s) in zip(levels, sc):
                self.trendNodeMatrix.loc[stock, level] = s
    
    def removeGaugeStockList(self, to_be_removed):
        self.trendNodeMatrix.drop(to_be_removed, inplace=True)
        self.stockList = list(self.trendNodeMatrix.index)
        
    def keepGaugeStockList(self, to_be_kept):
        self.trendNodeMatrix = self.trendNodeMatrix.loc[to_be_kept]
        self.stockList = to_be_kept
        
    def getGaugeStockList(self, stock_list):
        return self.trendNodeMatrix.loc[stock_list]
    
    def check_non_exists(self, stock_list): # find out stocks not in current list
        return [stock for stock in stock_list if stock not in self.stockList]
    
    def check_exists(self, stock_list):
        return [stock for stock in stock_list if stock in self.stockList]
    
    def appendStockList(self, stock_list_df):
        to_append = [stock for stock in stock_list_df.index if stock not in self.trendNodeMatrix.index]
        self.trendNodeMatrix=self.trendNodeMatrix.append(stock_list_df.loc[to_append], verify_integrity=True)
        self.stockList = list(self.trendNodeMatrix.index)
        
    def gaugeStock(self, stock, levels):
        gaugeList = []
        for level in levels:
            stock_df = SecurityDataManager.get_data_rq(stock, self.count, level, fields=['open','close','high','low'], skip_suspended=True)
            kb = KBarProcessor(stock_df, isdebug=False)
            gaugeList.append(kb.gaugeStatus(isSimple=False))
        return gaugeList

    def gaugeStock_analysis(self, stock, levels):
        print("retrieving data using get_price!!!")
        gaugeList = []
        for level in levels:
            today = datetime.datetime.today().date()
            start_date = get_trading_dates('2006-01-01', today)[-self.count]
            stock_df = SecurityDataManager.get_research_data_rq(stock, start_date=start_date, end_date=today, period=level, fields = ['open','close','high','low'], skip_suspended=True)
            kb = KBarProcessor(stock_df)
            gaugeList.append(kb.gaugeStatus(isSimple=False))
        return gaugeList
    
    def displayMonitorMatrix(self, stock_list=None):
        print(self.trendNodeMatrix.loc[stock_list] if stock_list else self.trendNodeMatrix)
                
    def filterLongPivotCombo(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(LongPivotCombo.matchStatus, stock_list, level_list, update_df)  
    
    def filterShortPivotCombo(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(ShortPivotCombo.matchStatus, stock_list, level_list, update_df)
    
    def filterLongStatusCombo(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(LongStatusCombo.matchStatus, stock_list, level_list, update_df)
    
    def filterShortStatusCombo(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(ShortStatusCombo.matchStatus, stock_list, level_list, update_df)
    
    
    def filterDownNodeDownNode(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(DownNodeDownNode.matchBiaoLiStatus, stock_list, level_list, update_df)
    
    def filterDownNodeUpTrend(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(DownNodeUpTrend.matchBiaoLiStatus, stock_list, level_list, update_df)

    def filterDownNodeDownTrend(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(DownNodeDownTrend.matchBiaoLiStatus, stock_list, level_list, update_df)  

    def filterDownNodeUpNode(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(DownNodeUpNode.matchBiaoLiStatus, stock_list, level_list, update_df)  


    def filterUpTrendDownNode(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(UpTrendDownNode.matchBiaoLiStatus, stock_list, level_list, update_df)

    def filterUpTrendUpTrend(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(UpTrendUpTrend.matchBiaoLiStatus, stock_list, level_list, update_df)
    
    def filterUpTrendUpNode(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(UpTrendUpNode.matchBiaoLiStatus, stock_list, level_list, update_df)

    def filterUpTrendDownTrend(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(UpTrendDownTrend.matchBiaoLiStatus, stock_list, level_list, update_df)
    
    
    def filterDownTrendDownTrend(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(DownTrendDownTrend.matchBiaoLiStatus, stock_list, level_list, update_df)    

    def filterDownTrendDownNode(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(DownTrendDownNode.matchBiaoLiStatus, stock_list, level_list, update_df)    

    def filterDownTrendUpTrend(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(DownTrendUpTrend.matchBiaoLiStatus, stock_list, level_list, update_df)    

    def filterDownTrendUpNode(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(DownTrendUpNode.matchBiaoLiStatus, stock_list, level_list, update_df)    


    def filterUpNodeDownTrend(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(UpNodeDownTrend.matchBiaoLiStatus, stock_list, level_list, update_df)
    
    def filterUpNodeDownNode(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(UpNodeDownNode.matchBiaoLiStatus, stock_list, level_list, update_df)
    
    def filterUpNodeUpTrend(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(UpNodeUpTrend.matchBiaoLiStatus, stock_list, level_list, update_df)

    def filterUpNodeUpNode(self, stock_list=None, level_list=None, update_df=False):
        return self.filterCombo_sup(UpNodeUpNode.matchBiaoLiStatus, stock_list, level_list, update_df)


    def filterCombo_sup(self, filter_method, stock_list=None, level_list=None, update_df=False):
        # two column per layer
        working_df = self.trendNodeMatrix.loc[stock_list] if stock_list else self.trendNodeMatrix # slice rows by input
        working_level = [l1 for l1 in ChanMatrix.gauge_level if l1 in level_list] if level_list else ChanMatrix.gauge_level
        for i in range(len(working_level)-1): #xrange
            if working_df.empty:
                break
            high_level = working_level[i]
            low_level = working_level[i+1]
            mask = working_df[[high_level,low_level]].apply(lambda x: filter_method(*x), axis=1)
            working_df = working_df[mask]
        if update_df:
            self.trendNodeMatrix = working_df
            self.stockList=list(working_df.index)
        return list(working_df.index)
    
    def filterByStatusAndLevel(self, level, status, stock_list=None, update_df=False): # level recursive filter 
        working_df = self.trendNodeMatrix.loc[stock_list] if stock_list else self.trendNodeMatrix
        if type(status) is list:
            working_df = working_df[working_df[level].isin(status)]
        else:
            working_df = working_df[working_df[level] == status]
        if update_df:
            self.trendNodeMatrix = working_df
            self.stockList = list(working_df.index)
        return list(working_df.index)
    
    def weeklyLongFilter(self):
        return self.filterByStatusAndLevel(level='5d', status=[KBarStatus.upTrend, KBarStatus.downTrendNode], update_df=True)
    
    def dailyLongFilter(self, stockList=None):
        return self.filterByStatusAndLevel(level='1d', status=[KBarStatus.upTrend, KBarStatus.downTrendNode], stock_list=stockList, update_df=False)
    
    def intraDayLongFilter(self, stockList=None):
        return self.filterByStatusAndLevel(level='30m', status=KBarStatus.downTrendNode, stock_list=stockList, update_df=False)
        
    def weeklyShortFilter(self, stockList=None):
        return self.filterByStatusAndLevel(level='5d', status=[KBarStatus.upTrendNode, KBarStatus.downTrend], stock_list=stockList, update_df=False)
    
    def weeklyShortDownTrend(self, stockList=None):
        return self.filterByStatusAndLevel(level='5d', status=KBarStatus.downTrend, stock_list=stockList, update_df=False)        
    
    def dailyShortFilter(self, stockList=None):
        return self.filterByStatusAndLevel(level='1d', status=[KBarStatus.upTrendNode, KBarStatus.downTrend], stock_list=stockList, update_df=False)
    
    def dailyShortUpTendNode(self, stockList=None):
        return self.filterByStatusAndLevel(level='1d', status=KBarStatus.upTrendNode, stock_list=stockList, update_df=False)

    def dailyShortdownTrend(self, stockList=None):
        return self.filterByStatusAndLevel(level='1d', status=KBarStatus.downTrend, stock_list=stockList, update_df=False)
            
    def intraDayShortFilter(self, stockList=None):
        return self.filterByStatusAndLevel(level='30m', status=KBarStatus.upTrendNode, stock_list=stockList, update_df=False)
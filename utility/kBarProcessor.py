# -*- encoding: utf8 -*-
'''
Created on 1 Aug 2017

@author: MetalInvest
'''
import numpy as np
import copy
from biaoLiStatus import * 

def synchOpenPrice(open, close, high, low):
    if open > close:
        return high
    else:
        return low

def synchClosePrice(open, close, high, low):
    if open < close:
        return high
    else:
        return low    

class KBarProcessor(object):
    '''
    This lib takes financial instrument data, and process it according the Chan(Zen) theory
    We need at least 100 K-bars in each input data set
    '''
    def __init__(self, kDf, isdebug=False):
        '''
        dataframe input must contain open, close, high, low columns
        '''
        self.isdebug = isdebug
        self.kDataFrame_origin = kDf
        self.kDataFrame_standardized = copy.deepcopy(kDf)
        self.kDataFrame_standardized = self.kDataFrame_standardized.assign(new_high=np.nan, new_low=np.nan, trend_type=np.nan)
        self.kDataFrame_marked = None
    
    def synchForChart(self):
        self.kDataFrame_standardized = self.kDataFrame_standardized.drop('new_high', 1)
        self.kDataFrame_standardized = self.kDataFrame_standardized.drop('new_low', 1)
        self.kDataFrame_standardized = self.kDataFrame_standardized.drop('trend_type', 1)
        self.kDataFrame_standardized['open'] = self.kDataFrame_standardized.apply(lambda row: synchOpenPrice(row['open'], row['close'], row['high'], row['low']), axis=1)
        self.kDataFrame_standardized['close'] = self.kDataFrame_standardized.apply(lambda row: synchClosePrice(row['open'], row['close'], row['high'], row['low']), axis=1)        
    
    def checkInclusive(self, first, second):
        # output: 0 = no inclusion, 1 = first contains second, 2 second contains first
        isInclusion = InclusionType.noInclusion
        first_high = first.high if np.isnan(first.new_high) else first.new_high
        second_high = second.high if np.isnan(second.new_high) else second.new_high
        first_low = first.low if np.isnan(first.new_low) else first.new_low
        second_low = second.low if np.isnan(second.new_low) else second.new_low
        
        if first_high <= second_high and first_low >= second_low:
            isInclusion = InclusionType.firstCsecond
        elif first_high >= second_high and first_low <= second_low:
            isInclusion = InclusionType.secondCfirst
        return isInclusion
    
    def isBullType(self, first, second): 
        # this is assuming first second aren't inclusive
        isBull = False
        if first.high < second.high:
            isBull = True
        return isBull
        
    def standardize(self):
        # 1. We need to make sure we start with first two K-bars without inclusive relationship
        # drop the first if there is inclusion, and check again
        while self.kDataFrame_standardized.shape[0] > 2:
            firstElem = self.kDataFrame_standardized.iloc[0]
            secondElem = self.kDataFrame_standardized.iloc[1]
            if self.checkInclusive(firstElem, secondElem) != InclusionType.noInclusion:
                self.kDataFrame_standardized.drop(self.kDataFrame_standardized.index[0], inplace=True)
                pass
            else:
                self.kDataFrame_standardized.ix[0,'new_high'] = firstElem.high
                self.kDataFrame_standardized.ix[0,'new_low'] = firstElem.low
                break

        # 2. loop through the whole data set and process inclusive relationship
        for idx in range(self.kDataFrame_standardized.shape[0]-2): # xrange
            currentElem = self.kDataFrame_standardized.iloc[idx]
            firstElem = self.kDataFrame_standardized.iloc[idx+1]
            secondElem = self.kDataFrame_standardized.iloc[idx+2]
            if self.checkInclusive(firstElem, secondElem) != InclusionType.noInclusion:
                trend = self.kDataFrame_standardized.ix[idx+1,'trend_type'] if not np.isnan(self.kDataFrame_standardized.ix[idx+1,'trend_type']) else self.isBullType(currentElem, firstElem)
                if trend:
                    self.kDataFrame_standardized.ix[idx+2,'new_high']=max(firstElem.high if np.isnan(firstElem.new_high) else firstElem.new_high, secondElem.high if np.isnan(secondElem.new_high) else secondElem.new_high)
                    self.kDataFrame_standardized.ix[idx+2,'new_low']=max(firstElem.low if np.isnan(firstElem.new_low) else firstElem.new_low, secondElem.low if np.isnan(secondElem.new_low) else secondElem.new_low)
                    pass
                else: 
                    self.kDataFrame_standardized.ix[idx+2,'new_high']=min(firstElem.high if np.isnan(firstElem.new_high) else firstElem.new_high, secondElem.high if np.isnan(secondElem.new_high) else secondElem.new_high)
                    self.kDataFrame_standardized.ix[idx+2,'new_low']=min(firstElem.low if np.isnan(firstElem.new_low) else firstElem.new_low, secondElem.low if np.isnan(secondElem.new_low) else secondElem.new_low)
                    pass
                self.kDataFrame_standardized.ix[idx+2,'trend_type']=trend
                self.kDataFrame_standardized.ix[idx+1,'new_high']=np.nan
                self.kDataFrame_standardized.ix[idx+1,'new_low']=np.nan
            else:
                if np.isnan(self.kDataFrame_standardized.ix[idx+1,'new_high']): 
                    self.kDataFrame_standardized.ix[idx+1,'new_high'] = firstElem.high 
                if np.isnan(self.kDataFrame_standardized.ix[idx+1,'new_low']): 
                    self.kDataFrame_standardized.ix[idx+1,'new_low'] = firstElem.low
                self.kDataFrame_standardized.ix[idx+2,'new_high'] = secondElem.high
                self.kDataFrame_standardized.ix[idx+2,'new_low'] = secondElem.low

        self.kDataFrame_standardized['high'] = self.kDataFrame_standardized['new_high']
        self.kDataFrame_standardized['low'] = self.kDataFrame_standardized['new_low']

        self.kDataFrame_standardized = self.kDataFrame_standardized[np.isfinite(self.kDataFrame_standardized['high'])]
        # lines below is for chart drawing
        if self.isdebug:
            self.synchForChart()
        return self.kDataFrame_standardized
    
    def checkTopBot(self, current, first, second):
        if first.high > current.high and first.high > second.high:
            return TopBotType.top
        elif first.low < current.low and first.low < second.low:
            return TopBotType.bot
        else:
            return TopBotType.noTopBot
        
    def markTopBot(self):
        self.kDataFrame_standardized = self.kDataFrame_standardized.assign(tb=TopBotType.noTopBot)
        # This function assume we have done the standardization process (no inclusion)
        for idx in range(self.kDataFrame_standardized.shape[0]-2): #xrange
            currentElem = self.kDataFrame_standardized.iloc[idx]
            firstElem = self.kDataFrame_standardized.iloc[idx+1]
            secondElem = self.kDataFrame_standardized.iloc[idx+2]
            topBotType = self.checkTopBot(currentElem, firstElem, secondElem)
            if topBotType != TopBotType.noTopBot:
                self.kDataFrame_standardized.ix[idx+1, 'tb'] = topBotType
        if self.isdebug:
            print(self.kDataFrame_standardized)

    def defineBi(self):
        self.kDataFrame_standardized = self.kDataFrame_standardized.assign(new_index=[i for i in range(len(self.kDataFrame_standardized))])
        working_df = self.kDataFrame_standardized[self.kDataFrame_standardized['tb']!=TopBotType.noTopBot]
#         print working_df
        currentStatus = firstStatus = TopBotType.noTopBot
        i = 0
        markedIndex = i + 1
        while i < working_df.shape[0]-1:
            currentFenXing = working_df.iloc[i]
            if markedIndex > working_df.shape[0]-1:
                break
            firstFenXing = working_df.iloc[markedIndex]
            
            currentStatus = TopBotType.bot if currentFenXing.tb == TopBotType.bot else TopBotType.top
            firstStatus = TopBotType.bot if firstFenXing.tb == TopBotType.bot else TopBotType.top
        
            if currentStatus == firstStatus:
                if currentStatus == TopBotType.top:
                    if currentFenXing['high'] < firstFenXing['high']:
                        working_df.ix[i,'tb'] = TopBotType.noTopBot
                        i = markedIndex
                        markedIndex = i+1
                        continue
                    else:
                        working_df.ix[markedIndex,'tb'] = TopBotType.noTopBot
                        i = markedIndex+1
                        markedIndex = i+1
                        continue
                elif currentStatus == TopBotType.bot:
                    if currentFenXing['low'] > firstFenXing['low']:
                        working_df.ix[i,'tb'] = TopBotType.noTopBot
                        i = markedIndex
                        markedIndex = i+1
                        continue
                    else:
                        working_df.ix[markedIndex,'tb'] = TopBotType.noTopBot
                        i = markedIndex+1
                        markedIndex = i+1
                        continue
            else: 
                # possible BI status 1 check top high > bot low 2 check more than 3 bars (strict BI) in between
                enoughKbarGap = (working_df.ix[markedIndex,'new_index'] - working_df.ix[i,'new_index']) >= 4
                if enoughKbarGap:
                    if currentStatus == TopBotType.top and currentFenXing['high'] > firstFenXing['low']:
                        pass
                    elif currentStatus == TopBotType.top and currentFenXing['high'] <= firstFenXing['low']:
                        working_df.ix[i,'tb'] = TopBotType.noTopBot
                    elif currentStatus == TopBotType.bot and currentFenXing['low'] < firstFenXing['high']:
                        pass
                    elif currentStatus == TopBotType.bot and currentFenXing['low'] >= firstFenXing['high']:
                        working_df.ix[i,'tb'] = TopBotType.noTopBot
                else:
                    working_df.ix[markedIndex,'tb'] = TopBotType.noTopBot
                    markedIndex += 1
                    # if marketIndex is the last one
                    if markedIndex > working_df.shape[0]-1 \
                    and ((working_df.ix[i,'tb']==TopBotType.top and self.kDataFrame_origin.ix[-1,'high'] > working_df.ix[i, 'high']) \
                    or (working_df.ix[i,'tb']==TopBotType.bot and self.kDataFrame_origin.ix[-1, 'low'] < working_df.ix[i, 'low']) ):
                        working_df.ix[i, 'tb'] = TopBotType.noTopBot
                    continue # don't increment i
            i+=1
            markedIndex = i+1
        self.kDataFrame_marked = working_df[working_df['tb']!=TopBotType.noTopBot]
        if self.isdebug:
            print(self.kDataFrame_marked)
    
    def getCurrentKBarStatus(self, isSimple=True):
        #  at Top or Bot FenXing
        resultStatus = KBarStatus.none_status
        
        # TODO, if not enough data given, long trend status can't be gauged here. We ignore it
        if self.kDataFrame_standardized.shape[0] < 2:
            return resultStatus
        
        if isSimple:
            if self.kDataFrame_standardized.ix[-2,'tb'] == TopBotType.top:
                resultStatus = KBarStatus.upTrendNode
            elif self.kDataFrame_standardized.ix[-2,'tb'] == TopBotType.bot:
                resultStatus = KBarStatus.downTrendNode
            elif self.kDataFrame_standardized.ix[-1, 'high'] > self.kDataFrame_standardized.ix[-2, 'high']:
                resultStatus = KBarStatus.upTrend
            elif self.kDataFrame_standardized.ix[-1, 'low'] < self.kDataFrame_standardized.ix[-2, 'low']:
                resultStatus = KBarStatus.downTrend
        else:
            if self.kDataFrame_marked.empty:
                return KBarStatus.upTrend if self.kDataFrame_standardized.ix[-1, 'high'] > self.kDataFrame_standardized.ix[-2, 'high'] else KBarStatus.downTrend
            
            if self.kDataFrame_marked.ix[-1, 'new_index'] == self.kDataFrame_standardized.shape[0]-2:
                if self.kDataFrame_marked.ix[-1,'tb'] == TopBotType.top:
                    resultStatus = KBarStatus.upTrendNode
                else:
                    resultStatus = KBarStatus.downTrendNode
            else:
                if self.kDataFrame_marked.ix[-1,'tb'] == TopBotType.top:
                    resultStatus = KBarStatus.downTrend
                else:
                    resultStatus = KBarStatus.upTrend
        return resultStatus
        
    def gaugeStatus(self, isSimple=True):
        self.standardize()
        self.markTopBot()
        if not isSimple:
            self.defineBi() # not necessary for biaoli status
        return self.getCurrentKBarStatus(isSimple)
    
    def getMarkedBL(self):
        self.standardize()
        self.markTopBot()
        self.defineBi()
        return self.kDataFrame_marked
        
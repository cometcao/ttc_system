# -*- encoding: utf8 -*-
'''
Created on 1 Aug 2017

@author: MetalInvest
'''
import numpy as np
import copy
from utility.biaoLiStatus import * 

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
    def __init__(self, kDf, isdebug=False, clean_standardzed=False):
        '''
        dataframe input must contain open, close, high, low columns
        '''
        self.isdebug = isdebug
        self.clean_standardzed = clean_standardzed
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
            inclusion_type = self.checkInclusive(firstElem, secondElem)
            if inclusion_type != InclusionType.noInclusion:
                trend = self.kDataFrame_standardized.ix[idx+1,'trend_type'] if not np.isnan(self.kDataFrame_standardized.ix[idx+1,'trend_type']) else self.isBullType(currentElem, firstElem)
                compare_func = max if trend else min
                first_index = idx+1
                second_index = idx+2
                if inclusion_type == InclusionType.firstCsecond:
                    self.kDataFrame_standardized.ix[second_index,'new_high']=compare_func(firstElem.high if np.isnan(firstElem.new_high) else firstElem.new_high, secondElem.high if np.isnan(secondElem.new_high) else secondElem.new_high)
                    self.kDataFrame_standardized.ix[second_index,'new_low']=compare_func(firstElem.low if np.isnan(firstElem.new_low) else firstElem.new_low, secondElem.low if np.isnan(secondElem.new_low) else secondElem.new_low)
                    self.kDataFrame_standardized.ix[second_index,'trend_type']=trend
                    self.kDataFrame_standardized.ix[first_index,'new_high']=np.nan
                    self.kDataFrame_standardized.ix[first_index,'new_low']=np.nan   
                else:                 
                    self.kDataFrame_standardized.ix[first_index,'new_high']=compare_func(firstElem.high if np.isnan(firstElem.new_high) else firstElem.new_high, secondElem.high if np.isnan(secondElem.new_high) else secondElem.new_high)
                    self.kDataFrame_standardized.ix[first_index,'new_low']=compare_func(firstElem.low if np.isnan(firstElem.new_low) else firstElem.new_low, secondElem.low if np.isnan(secondElem.new_low) else secondElem.new_low)                        
                    self.kDataFrame_standardized.ix[first_index,'trend_type']=trend
                    self.kDataFrame_standardized.ix[second_index,'new_high']=np.nan
                    self.kDataFrame_standardized.ix[second_index,'new_low']=np.nan                
            else:
                if np.isnan(self.kDataFrame_standardized.ix[idx+1,'new_high']): 
                    self.kDataFrame_standardized.ix[idx+1,'new_high'] = firstElem.high 
                if np.isnan(self.kDataFrame_standardized.ix[idx+1,'new_low']): 
                    self.kDataFrame_standardized.ix[idx+1,'new_low'] = firstElem.low
                if np.isnan(self.kDataFrame_standardized.ix[idx+2,'new_high']): 
                    self.kDataFrame_standardized.ix[idx+2,'new_high'] = secondElem.high
                if np.isnan(self.kDataFrame_standardized.ix[idx+2,'new_low']): 
                    self.kDataFrame_standardized.ix[idx+2,'new_low'] = secondElem.low

        self.kDataFrame_standardized['high'] = self.kDataFrame_standardized['new_high']
        self.kDataFrame_standardized['low'] = self.kDataFrame_standardized['new_low']

        self.kDataFrame_standardized = self.kDataFrame_standardized[np.isfinite(self.kDataFrame_standardized['high'])]
        # lines below is for chart drawing
        if self.clean_standardzed:
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
        
#         ########### clean up zigzag topbot #############
#         working_df['new_index_offset'] = working_df['new_index'].shift(-1) - working_df['new_index']
#         loop_index = 0
#         while loop_index < working_df.shape[0]-3:
#             previous_idx = loop_index
#             current_idx = loop_index+1
#             next_idx = loop_index+2
#             
#             if working_df.ix[previous_idx,'new_index_offset'] < 4: # strict Bi definition
#                 if working_df.ix[current_idx, 'tb'] == TopBotType.top:
#                     if working_df.ix[previous_idx, 'low'] < working_df.ix[next_idx, 'low']:
#                         working_df.ix[current_idx, 'tb'] = TopBotType.noTopBot
#                     else:
#                         working_df.ix[previous_idx, 'tb'] = TopBotType.noTopBot
#                 else: # current bot
#                     if working_df.ix[previous_idx, 'high'] > working_df.ix[next_idx, 'high']:
#                         working_df.ix[current_idx, 'tb'] = TopBotType.noTopBot
#                     else:
#                         working_df.ix[previous_idx, 'tb'] = TopBotType.noTopBot
#             loop_index += 1
# 
#         working_df.drop('new_index_offset', axis = 1, inplace=True)
#         working_df = working_df[working_df['tb']!=TopBotType.noTopBot]
        ################################################
        

        ############################# clean up the base case for Bi definition ###########################
        firstIdx = 0
        while firstIdx < working_df.shape[0]:
            firstFenXing = working_df.iloc[firstIdx]
            secondFenXing = working_df.iloc[firstIdx+1]
            if firstFenXing.tb == secondFenXing.tb:
                firstFenXing.tb = TopBotType.noTopBot
            elif secondFenXing.new_index - firstFenXing.new_index < 4:
                firstFenXing.tb = TopBotType.noTopBot
            elif firstFenXing.tb == TopBotType.top and secondFenXing.tb == TopBotType.bot and firstFenXing.high <= secondFenXing.high:
                firstFenXing.tb = TopBotType.noTopBot
            elif firstFenXing.tb == TopBotType.bot and secondFenXing.tb == TopBotType.top and firstFenXing.low >= secondFenXing.low:
                firstFenXing.tb = TopBotType.noTopBot
            else:
                break
            firstIdx += 1
        #################################
#         print (working_df)
        working_df = working_df[working_df['tb']!=TopBotType.noTopBot]
        previousStatus = currentStatus = nextStatus = TopBotType.noTopBot
        previous_index = 0
        current_index = previous_index + 1
        next_index = current_index + 1
        #################################
        while previous_index < working_df.shape[0]-2 and current_index < working_df.shape[0]-1 and next_index < working_df.shape[0]:
            previousFenXing = working_df.iloc[previous_index]
            currentFenXing = working_df.iloc[current_index]
            nextFenXing = working_df.iloc[next_index]
            
            previousStatus = previousFenXing.tb
            currentStatus = currentFenXing.tb
            nextStatus = nextFenXing.tb

            if currentStatus == previousStatus:
                if currentStatus == TopBotType.top:
                    if currentFenXing['high'] < previousFenXing['high']:
                        working_df.ix[current_index,'tb'] = TopBotType.noTopBot
                        current_index = next_index
                        next_index = current_index+1
                        continue
                    else:
                        working_df.ix[previous_index,'tb'] = TopBotType.noTopBot
                        previous_index = current_index
                        current_index = next_index
                        next_index += 1
                        continue
                elif currentStatus == TopBotType.bot:
                    if currentFenXing['low'] > previousFenXing['low']:
                        working_df.ix[current_index,'tb'] = TopBotType.noTopBot
                        current_index = next_index
                        next_index = current_index+1
                        continue
                    else:
                        working_df.ix[previous_index,'tb'] = TopBotType.noTopBot
                        previous_index = current_index
                        current_index = next_index
                        next_index += 1
                        continue

            # possible BI status 1 check top high > bot low 2 check more than 3 bars (strict BI) in between
            enoughKbarGap = (working_df.ix[next_index,'new_index'] - working_df.ix[current_index,'new_index']) >= 4
            if enoughKbarGap:
                if currentStatus == TopBotType.top and currentFenXing['high'] > nextFenXing['high']:
                    pass
                elif currentStatus == TopBotType.top and currentFenXing['high'] <= nextFenXing['high']:
                    working_df.ix[current_index,'tb'] = TopBotType.noTopBot
                    current_index = next_index
                    next_index = next_index + 1
                    continue
                elif currentStatus == TopBotType.bot and currentFenXing['low'] < nextFenXing['low']:
                    pass
                elif currentStatus == TopBotType.bot and currentFenXing['low'] >= nextFenXing['low']:
                    working_df.ix[current_index,'tb'] = TopBotType.noTopBot   
                    current_index = next_index 
                    next_index = next_index + 1
                    continue
            else: # not enough gaps this shouldn't happen, but we have precautions
                if currentStatus == TopBotType.top and previousFenXing['low'] < nextFenXing['low']:
                    working_df.ix[next_index,'tb'] = TopBotType.noTopBot
                    next_index += 1
                    continue
                if currentStatus == TopBotType.top and previousFenXing['low'] >= nextFenXing['low']:
                    working_df.ix[current_index,'tb'] = TopBotType.noTopBot
                    current_index = next_index
                    next_index += 1
                    continue
                if currentStatus == TopBotType.bot and previousFenXing['high'] > nextFenXing['high']:
                    working_df.ix[next_index,'tb'] = TopBotType.noTopBot
                    next_index += 1
                    continue              
                if currentStatus == TopBotType.bot and previousFenXing['high'] <= nextFenXing['high']:
                    working_df.ix[current_index,'tb'] = TopBotType.noTopBot
                    current_index = next_index
                    next_index += 1
                    continue                          

            previous_index = current_index
            current_index=next_index
            next_index = current_index+1
            
            # if nextIndex is the last one
            if next_index == working_df.shape[0]-1 \
            and ((working_df.ix[current_index,'tb']==TopBotType.top and self.kDataFrame_origin.ix[-1,'high'] > working_df.ix[current_index, 'high']) \
            or (working_df.ix[current_index,'tb']==TopBotType.bot and self.kDataFrame_origin.ix[-1, 'low'] < working_df.ix[current_index, 'low']) ):
                working_df.ix[current_index, 'tb'] = TopBotType.noTopBot
                
        ###################################    
        self.kDataFrame_marked = working_df[working_df['tb']!=TopBotType.noTopBot]
        if self.isdebug:
            print(self.kDataFrame_marked)

    def defineBi_old(self):
        self.kDataFrame_standardized = self.kDataFrame_standardized.assign(new_index=[i for i in range(len(self.kDataFrame_standardized))])
        working_df = self.kDataFrame_standardized[self.kDataFrame_standardized['tb']!=TopBotType.noTopBot]
        print (working_df)
        currentStatus = nextStatus = TopBotType.noTopBot
        current_index = 0
        next_index = current_index + 1
        while current_index < working_df.shape[0]-1:
            currentFenXing = working_df.iloc[current_index]
            if next_index > working_df.shape[0]-1:
                break
            nextFenXing = working_df.iloc[next_index]
            
            currentStatus = TopBotType.bot if currentFenXing.tb == TopBotType.bot else TopBotType.top
            nextStatus = TopBotType.bot if nextFenXing.tb == TopBotType.bot else TopBotType.top
        
            if currentStatus == nextStatus:
                if currentStatus == TopBotType.top:
                    if currentFenXing['high'] <= nextFenXing['high']:
                        working_df.ix[current_index,'tb'] = TopBotType.noTopBot
                        current_index = next_index
                        next_index = current_index+1
                        continue
                    else:
                        working_df.ix[next_index,'tb'] = TopBotType.noTopBot
                        next_index += 1
                        continue
                elif currentStatus == TopBotType.bot:
                    if currentFenXing['low'] >= nextFenXing['low']:
                        working_df.ix[current_index,'tb'] = TopBotType.noTopBot
                        current_index = next_index
                        next_index = current_index+1
                        continue
                    else:
                        working_df.ix[next_index,'tb'] = TopBotType.noTopBot
                        next_index += 1
                        continue
            else: 
                # possible BI status 1 check top high > bot low 2 check more than 3 bars (strict BI) in between
                enoughKbarGap = (working_df.ix[next_index,'new_index'] - working_df.ix[current_index,'new_index']) >= 4
                if enoughKbarGap:
                    if currentStatus == TopBotType.top and currentFenXing['high'] > nextFenXing['high']:
                        pass
                    elif currentStatus == TopBotType.top and currentFenXing['high'] <= nextFenXing['high']:
                        working_df.ix[current_index,'tb'] = TopBotType.noTopBot
                    elif currentStatus == TopBotType.bot and currentFenXing['low'] < nextFenXing['low']:
                        pass
                    elif currentStatus == TopBotType.bot and currentFenXing['low'] >= nextFenXing['low']:
                        working_df.ix[current_index,'tb'] = TopBotType.noTopBot
                else:
                    working_df.ix[next_index,'tb'] = TopBotType.noTopBot
                    next_index += 1
                    # if nextIndex is the last one
                    if next_index > working_df.shape[0]-1 \
                    and ((working_df.ix[current_index,'tb']==TopBotType.top and self.kDataFrame_origin.ix[-1,'high'] > working_df.ix[current_index, 'high']) \
                    or (working_df.ix[current_index,'tb']==TopBotType.bot and self.kDataFrame_origin.ix[-1, 'low'] < working_df.ix[current_index, 'low']) ):
                        working_df.ix[current_index, 'tb'] = TopBotType.noTopBot
                    continue # don't increment current_index
            current_index=next_index
            next_index = current_index+1
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
    
    def getStandardized(self):
        return self.standardize()
    
    def getIntegraded(self):
        self.standardize()
        self.markTopBot()
        self.defineBi()
        return self.kDataFrame_origin.join(self.kDataFrame_marked[['new_index', 'tb']])
        
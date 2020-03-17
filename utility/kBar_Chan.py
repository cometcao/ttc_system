# -*- encoding: utf8 -*-

import numpy as np
import copy
import talib
from utility.biaoLiStatus import * 
from utility.chan_common_include import GOLDEN_RATIO, MIN_PRICE_UNIT
from kBarProcessor import synchOpenPrice, synchClosePrice
from numpy.lib.recfunctions import append_fields


def get_previous_loc(loc, working_df):
    i = loc - 1
    while i >= 0:
        if working_df[i]['tb'] == TopBotType.top or working_df[i]['tb'] == TopBotType.bot:
            return i
        else:
            i = i - 1
    return None

def get_next_loc(loc, working_df):
    i = loc + 1
    while i < len(working_df):
        if working_df[i]['tb'] == TopBotType.top or working_df[i]['tb'] == TopBotType.bot:
            return i
        else:
            i = i + 1
    return None


class KBarChan(object):
    '''
    This is a rewrite of KBarProcessor, that one is too slow!! we used pandas dataframe, we should use numpy array!
    df=False flag is used here
    '''
    
    def __init__(self, kDf, isdebug=False, clean_standardzed=False):
        self.isdebug = isdebug
        self.clean_standardzed = clean_standardzed
        self.kDataFrame_origin = kDf
        self.kDataFrame_standardized = copy.deepcopy(kDf)
        self.kDataFrame_standardized = append_fields(self.kDataFrame_standardized, 
                                                     ['new_high', 'new_low', 'trend_type', 'real_loc'],
                                                     [
                                                         [0]*len(self.kDataFrame_standardized),
                                                         [0]*len(self.kDataFrame_standardized),
                                                         [TopBotType.noTopBot]*len(self.kDataFrame_standardized),
                                                         [i for i in range(len(self.kDataFrame_standardized))]
                                                     ],
                                                     usemask=False)
        self.kDataFrame_marked = None
        self.kDataFrame_xd = None
        self.gap_XD = []
        self.previous_with_xd_gap = False # help to check current gap as XD
        
    def checkInclusive(self, first, second):
        # output: 0 = no inclusion, 1 = first contains second, 2 second contains first
        isInclusion = InclusionType.noInclusion
        first_high = first['high'] if first['new_high']==0 else first['new_high']
        second_high = second['high'] if second['new_high']==0 else second['new_high']
        first_low = first['low'] if first['new_low']==0 else first['new_low']
        second_low = second['low'] if second['new_low']==0 else second['new_low']
        
        if first_high <= second_high and first_low >= second_low:
            isInclusion = InclusionType.firstCsecond
        elif first_high >= second_high and first_low <= second_low:
            isInclusion = InclusionType.secondCfirst
        return isInclusion
    
    def isBullType(self, first, second): 
        # this is assuming first second aren't inclusive
        isBull = False
        f_high = first['high'] if first['new_high']==0 else first['new_high']
        s_high = second['high'] if second['new_high']==0 else second['new_high']
        if f_high < s_high:
            isBull = True
        return isBull
        
    def standardize(self, initial_state=TopBotType.noTopBot):
        # 1. We need to make sure we start with first two K-bars without inclusive relationship
        # drop the first if there is inclusion, and check again
        if initial_state == TopBotType.top or initial_state == TopBotType.bot:
            # given the initial state, make the first two bars non-inclusive, 
            # the first bar is confirmed as pivot, anything followed with inclusive relation 
            # will be merged into the first bar
            while len(self.kDataFrame_standardized) > 2:
                first_Elem = self.kDataFrame_standardized[0]
                second_Elem = self.kDataFrame_standardized[1]
                if self.checkInclusive(first_Elem, second_Elem) != InclusionType.noInclusion:
                    if initial_state == TopBotType.bot:
                        self.kDataFrame_standardized[0]['new_high'] = second_Elem['high']
                        self.kDataFrame_standardized[0]['new_low'] = first_Elem['low']
                    elif initial_state == TopBotType.top:
                        self.kDataFrame_standardized[0]['new_high'] = first_Elem['high']
                        self.kDataFrame_standardized[0]['new_low'] = second_Elem['low']
                    self.kDataFrame_standardized=np.delete(self.kDataFrame_standardized, 1, axis=0)
                else:
                    self.kDataFrame_standardized[0]['new_high'] = first_Elem['high']
                    self.kDataFrame_standardized[0]['new_low'] = first_Elem['low']
                    break                    
        else:
            while len(self.kDataFrame_standardized) > 2:
                first_Elem = self.kDataFrame_standardized[0]
                second_Elem = self.kDataFrame_standardized[1]
                if self.checkInclusive(first_Elem, second_Elem) != InclusionType.noInclusion:
                    self.kDataFrame_standardized=np.delete(self.kDataFrame_standardized, 0, axis=0)
                else:
                    self.kDataFrame_standardized[0]['new_high'] = first_Elem['high']
                    self.kDataFrame_standardized[0]['new_low'] = first_Elem['low']
                    break


        # 2. loop through the whole data set and process inclusive relationship
        pastElemIdx = 0
        firstElemIdx = pastElemIdx+1
        secondElemIdx = firstElemIdx+1
        high='high'
        low='low'
        new_high = 'new_high'
        new_low = 'new_low'
        trend_type = 'trend_type'
        
        while secondElemIdx < len(self.kDataFrame_standardized): # xrange
            pastElem = self.kDataFrame_standardized[pastElemIdx]
            firstElem = self.kDataFrame_standardized[firstElemIdx]
            secondElem = self.kDataFrame_standardized[secondElemIdx]
            inclusion_type = self.checkInclusive(firstElem, secondElem)
            if inclusion_type != InclusionType.noInclusion:
                trend = firstElem[trend_type] if firstElem[trend_type]!=TopBotType.noTopBot else self.isBullType(pastElem, firstElem)
                compare_func = max if trend==TopBotType.bot2top else min
                if inclusion_type == InclusionType.firstCsecond:
                    self.kDataFrame_standardized[secondElemIdx][new_high]=compare_func(firstElem[high] if firstElem[new_high]==0 else firstElem[new_high], secondElem[high] if secondElem[new_high]==0 else secondElem[new_high])
                    self.kDataFrame_standardized[secondElemIdx][new_low]=compare_func(firstElem[low] if firstElem[new_low]==0 else firstElem[new_low], secondElem[low] if secondElem[new_low]==0 else secondElem[new_low])
                    self.kDataFrame_standardized[secondElemIdx][trend_type] = trend
                    self.kDataFrame_standardized[firstElemIdx][new_high]=0
                    self.kDataFrame_standardized[firstElemIdx][new_low]=0
                    ############ manage index for next round ###########
                    firstElemIdx = secondElemIdx
                    secondElemIdx += 1
                else:
                    self.kDataFrame_standardized[firstElemIdx][new_high]=compare_func(firstElem[high] if firstElem[new_high]==0 else firstElem[new_high], secondElem[high] if secondElem[new_high]==0 else secondElem[new_high])
                    self.kDataFrame_standardized[firstElemIdx][new_low]=compare_func(firstElem[low] if firstElem[new_low]==0 else firstElem[new_low], secondElem[low] if secondElem[new_low]==0 else secondElem[new_low])                        
                    self.kDataFrame_standardized[firstElemIdx][trend_type]=trend
                    self.kDataFrame_standardized[secondElemIdx][new_high]=0
                    self.kDataFrame_standardized[secondElemIdx][new_low]=0
                    ############ manage index for next round ###########
                    secondElemIdx += 1
            else:
                if np.isnan(firstElem[new_high]): 
                    self.kDataFrame_standardized[firstElemIdx][new_high] = firstElem[high]
                if np.isnan(firstElem[new_low]): 
                    self.kDataFrame_standardized[firstElemIdx][new_low] = firstElem[low]
                if np.isnan(secondElem[new_high]): 
                    self.kDataFrame_standardized[secondElemIdx][new_high] = secondElem[high]
                if np.isnan(secondElem[new_low]): 
                    self.kDataFrame_standardized[secondElemIdx][new_low] = secondElem[low]
                ############ manage index for next round ###########
                pastElemIdx = firstElemIdx
                firstElemIdx = secondElemIdx
                secondElemIdx += 1
                
        # clean up
        self.kDataFrame_standardized[high] = self.kDataFrame_standardized[new_high]
        self.kDataFrame_standardized[low] = self.kDataFrame_standardized[new_low]
        self.kDataFrame_standardized=self.kDataFrame_standardized[['close', 'high', 'low', 'real_loc']]

        # remove standardized kbars
        self.kDataFrame_standardized = self.kDataFrame_standardized[self.kDataFrame_standardized['high']!=0]

        # new index add for later distance calculation => straight after standardization
        self.kDataFrame_standardized = append_fields(self.kDataFrame_standardized, 
                                                     'new_index',
                                                     [i for i in range(len(self.kDataFrame_standardized))],
                                                     usemask=False)
        return self.kDataFrame_standardized
    
    def checkTopBot(self, current, first, second):
        if first['high'] > current['high'] and first['high'] > second['high']:
            return TopBotType.top
        elif first['low'] < current['low'] and first['low'] < second['low']:
            return TopBotType.bot
        else:
            return TopBotType.noTopBot
    
    def markTopBot(self, initial_state=TopBotType.noTopBot):
        self.kDataFrame_standardized = append_fields(self.kDataFrame_standardized,
                                                     'tb',
                                                     [TopBotType.noTopBot]*self.kDataFrame_standardized.size,
                                                     usemask=False
                                                     )
        if self.kDataFrame_standardized.size < 7:
            return
        tb = 'tb'
        if initial_state == TopBotType.top or initial_state == TopBotType.bot:
            felem = self.kDataFrame_standardized[0]
            selem = self.kDataFrame_standardized[1]
            if (initial_state == TopBotType.top and felem['high'] >= selem['high']) or \
                (initial_state == TopBotType.bot and felem['low'] <= selem['low']):
                self.kDataFrame_standardized[0][tb] = initial_state
            else:
                if self.isdebug:
                    print("Incorrect initial state given!!!")
                
        # This function assume we have done the standardization process (no inclusion)
        last_idx = 0
        for idx in range(self.kDataFrame_standardized.size-2): #xrange
            currentElem = self.kDataFrame_standardized[idx]
            firstElem = self.kDataFrame_standardized[idx+1]
            secondElem = self.kDataFrame_standardized[idx+2]
            topBotType = self.checkTopBot(currentElem, firstElem, secondElem)
            if topBotType != TopBotType.noTopBot:
                self.kDataFrame_standardized[idx+1][tb] = topBotType
                last_idx = idx+1
                
        # mark the first kbar
        if (self.kDataFrame_standardized[0][tb] != TopBotType.top and\
            self.kDataFrame_standardized[0][tb] != TopBotType.bot):
            first_loc = get_next_loc(0, self.kDataFrame_standardized)
            if first_loc is not None:
                first_tb = self.kDataFrame_standardized[first_loc][tb]
                self.kDataFrame_standardized[0][tb] = TopBotType.reverse(first_tb)
            
        # mark the last kbar 
        last_tb = self.kDataFrame_standardized[last_idx][tb]
        self.kDataFrame_standardized[-1][tb] = TopBotType.reverse(last_tb)
        if self.isdebug:
            print("self.kDataFrame_standardized[20]:{0}".format(self.kDataFrame_standardized[:20]))
            
    def trace_back_index(self, working_df, previous_index):
        # find the closest FenXing with top/bot backwards from previous_index
        idx = previous_index-1
        while idx >= 0:
            fx = working_df[idx]
            if fx['tb'] == TopBotType.noTopBot:
                idx -= 1
                continue
            else:
                return idx
        if self.isdebug:
            print("We don't have previous valid FenXing")
        return None
    
    def prepare_original_kdf(self):
        _, _, macd = talib.MACD(self.kDataFrame_origin['close'].values)
        self.kDataFrame_origin = append_fields(self.kDataFrame_standardized,
                                                'macd',
                                                macd,
                                                usemask=False)

    def gap_exists_in_range(self, start_idx, end_idx): # end_idx included
        gap_working_df = self.kDataFrame_origin[start_idx:end_idx, :]
        
        # drop the first row, as we only need to find gaps from start_idx(non-inclusive) to end_idx(inclusive)        
        gap_working_df.drop(gap_working_df.index[0], inplace=True)
        return len(gap_working_df[gap_working_df['gap']==True]) > 0
    
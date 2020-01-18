# -*- encoding: utf8 -*-
'''
Created on 18 Jan 2020

@author: MetalInvest
'''
import numpy as np
from utility.centralRegion import Chan_Type
from utility.biaoLiStatus import TopBotType
from utility.securityDataManager import *

TYPE_III_NUM = 5
TYPE_I_NUM = 10

def filter_high_level_by_index(period = '5d',direction=TopBotType.top2bot, stock_index='000985.XSHG', df=False):
    all_stocks = JqDataRetriever.get_index_stocks(stock_index)
    result_stocks = []
    for stock in all_stocks:
        stock_high = JqDataRetriever.get_bars(stock, 
                                               count=max(TYPE_I_NUM, TYPE_III_NUM), 
                                               end_dt=pd.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                               unit=period,
                                               fields= ['open',  'high', 'low','close', 'money'], 
                                               df = df)
        if KBar.filter_high_level_kbar(stock_high, direction=direction, df=df, chan_type=Chan_Type.III):
            result_stocks.append(stock)
        if KBar.filter_high_level_kbar(stock_high, direction=direction, df=df, chan_type=Chan_Type.I):
            result_stocks.append(stock)
    print("qualifying stocks:{0}".format(result_stocks))
    
    return result_stocks

class KBar(object):
    '''
    used for initial filter with OCHL
    '''
    def __init__(self, open, close, high, low):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        
    def __repr__(self):
        return "\nOPEN:{0} CLOSE:{1} HIGH:{2} LOW:{3}".format(self.open, self.close, self.high, self.low)

    @classmethod
    def contain_zhongshu(cls, first, second, third, return_core_range=False):
        '''
        check consecutive three kbars with intersection, 
        return_core_range controls return range type, default return core range to increase the robostness of initial filters
        TYPE III: False
        TYPE I : True
        '''
        if first.high >= second.high >= first.low or\
            first.high >= second.low >= first.low or\
            second.high >= first.high >= second.low or\
            second.high >= first.low >= second.low:
            new_high, new_low = min(first.high, second.high), max(first.low, second.low)
            if new_high >= third.high >= new_low or\
                new_high >= third.low >= new_low or\
                third.high >= new_high >= third.low or\
                third.high >= new_low >= third.low:
                if return_core_range:
                    return True, min(first.high, second.high, third.high), max(first.low, second.low, third.low)
                else:
                    return True, max(first.high, second.high, third.high), min(first.low, second.low, third.low)
        return False, 0, 0
    
    @classmethod 
    def create_N_kbar(cls, high_df, n = TYPE_III_NUM):
        i = -n
        kbar_list = []
        while i <= -1:
            kb = KBar(high_df['open'][i], high_df['close'][i], high_df['high'][i], high_df['low'][i])
            kbar_list.append(kb)
            i = i + 1
        return kbar_list
        
        
    @classmethod
    def chan_type_III_check(cls, kbar_list, direction):
        '''
        max 5 high level KBars
        '''
        first = kbar_list[-TYPE_III_NUM]
        second = kbar_list[-TYPE_III_NUM+1] 
        third = kbar_list[-TYPE_III_NUM+2]
        forth = kbar_list[-TYPE_III_NUM+3]
        fifth = kbar_list[-TYPE_III_NUM+4]
        result = False
        if fifth.close < fifth.open or fifth.close < (fifth.high + fifth.low)/2.0:
            check_result, k_m, k_l = cls.contain_zhongshu(first, second, third, return_core_range=False)
            if check_result:
                result = fifth.low > k_m if direction == TopBotType.top2bot else fifth.high < k_l
            if not result:
                check_result, k_m, k_l = cls.contain_zhongshu(second, third, forth, return_core_range=False)
                result = check_result and fifth.close > k_m if direction == TopBotType.top2bot else fifth.high < k_l 
        return result
    
    @classmethod
    def chan_type_I_check(cls, kbar_list, direction):
        '''
        Max number of high level Kbars
        '''
        result = False
        i = 0
        first_zs = False
        first_zs_ma = 0
        first_zs_mi = 0
        while i+3 < len(kbar_list):
            first = kbar_list[i]
            second = kbar_list[i+1]
            third = kbar_list[i+2]
            check_result, ma, mi = cls.contain_zhongshu(first, second, third, return_core_range=True)
            if not first_zs:
                if check_result:
                    first_zs = True
                    first_zs_ma = ma
                    first_zs_mi = mi
                    i = i + 3
                else:
                    i = i + 1
            else:
                if check_result:
                    result = first_zs_ma < mi if direction == TopBotType.bot2top else first_zs_mi > ma
                    if result:
                        break
                i = i + 1
        return result

    @classmethod
    def filter_high_level_kbar(cls, high_df, direction=TopBotType.top2bot, df=True, chan_type=Chan_Type.III):
        '''
        This method used by weekly (5d) data to find out rough 5m Zhongshu and 
        type III trade point
        
        It is used as initial filters for all stocks on markets
        
        1. find nearest 5m zhongshu => three consecutive weekly kbar share price region
        2. the kbar following the zhongshu escapes by direction, 
        3. the same kbar or next kbar's return attempt never touch the previous zhongshu (only strong case)
        '''
        result = False
        num_of_kbar = TYPE_III_NUM if chan_type == Chan_Type.III else TYPE_I_NUM
        if df:
            if high_df.shape[0] >= 4:
                i = -num_of_kbar
                kbar_list = []
                while i <= -1:
                    item = high_df.iloc[i]
                    kbar_list.append(item)
                    i = i + 1
                if chan_type == Chan_Type.III:
                    result = cls.chan_type_III_check(kbar_list, direction)
                elif chan_type == Chan_Type.I:
                    result = cls.chan_type_I_check(kbar_list, direction)
        else:
            if len(high_df['open']) >= 4:
                kbar_list = cls.create_N_kbar(high_df, n=num_of_kbar)
                if chan_type == Chan_Type.III:
                    result = cls.chan_type_III_check(kbar_list, direction)
                elif chan_type == Chan_Type.I:
                    result = cls.chan_type_I_check(kbar_list, direction)          
        return result 
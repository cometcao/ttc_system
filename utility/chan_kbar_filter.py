# -*- encoding: utf8 -*-
'''
Created on 18 Jan 2020

@author: MetalInvest
'''
import numpy as np
import pandas as pd

from utility.biaoLiStatus import TopBotType
from utility.securityDataManager import *

from utility.chan_common_include import *

def filter_high_level_by_index(direction=TopBotType.top2bot, 
                               stock_index=['000985.XSHG'], 
                               df=False, 
                               periods = ['60m', '120m', '1d'],
                               end_dt=pd.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                               chan_types=[Chan_Type.I, Chan_Type.III]):
    all_stocks = []
    for idx in stock_index:
        idx_stocks = JqDataRetriever.get_index_stocks(idx, end_dt)
        all_stocks = all_stocks + idx_stocks
    
    all_stocks = list(set(all_stocks))
    
    return filter_high_level_by_stocks(
                                        direction,
                                        all_stocks,
                                        df=df,
                                        periods=periods,
                                        end_dt=end_dt,
                                        chan_types=chan_types
                                        )

def filter_high_level_by_stocks(direction, 
                               stock_list, 
                               df=False, 
                               periods = ['30m'],
                               end_dt=pd.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                               chan_types=[Chan_Type.I, Chan_Type.III]):
    
    result_stocks_I = set()
    result_stocks_III = set()
    result_stocks_PB = set()
    for stock in stock_list:
        for p in periods:
            result_stocks_I, result_stocks_III, result_stocks_PB = filter_high_level_by_stock(stock, 
                                                                            p, 
                                                                            end_dt, 
                                                                            df, 
                                                                            chan_types,
                                                                            direction,
                                                                            result_stocks_I, 
                                                                            result_stocks_III,
                                                                            result_stocks_PB)
    type_i = sorted(list(result_stocks_I))
    type_iii = sorted(list(result_stocks_III))
    type_pb = sorted(list(result_stocks_PB))
    print("qualifying type I stocks:{0} {1} \ntype III stocks: {2} {3} \ntype PB stocks: {4} {5}".format(type_i,
                                                                                                         len(type_i),
                                                                                             type_iii,
                                                                                             len(type_iii),
                                                                                             type_pb,
                                                                                             len(type_pb)))
    
    return sorted(list(set(type_i + type_iii + type_pb)))
    

def filter_high_level_by_stock(stock, 
                               period, 
                               end_dt, 
                               df, 
                               chan_types, 
                               direction,
                               result_stocks_I, 
                               result_stocks_III,
                               result_stocks_PB):
    stock_high = JqDataRetriever.get_bars(stock, 
                                           count=max(TYPE_I_NUM, TYPE_III_NUM), 
                                           end_dt=end_dt, 
                                           unit=period,
                                           fields= ['open',  'high', 'low','close'], 
                                           df = df,
                                           include_now=True)
#     print("working on stock: {0}".format(stock))
    chan_type_results = KBar.filter_high_level_kbar(stock_high, 
                                   direction=direction, 
                                   df=df, 
                                   chan_types=chan_types)
    if Chan_Type.I in chan_type_results or Chan_Type.I_weak in chan_type_results:
        result_stocks_I.add(stock)
    elif Chan_Type.III in chan_type_results or Chan_Type.III_strong in chan_type_results:
        result_stocks_III.add(stock)
    elif Chan_Type.INVALID in chan_type_results:
        result_stocks_PB.add(stock)
    
    return result_stocks_I, result_stocks_III, result_stocks_PB

LONG_MA_NUM = 13
SHORT_MA_NUM = 5

def analyze_MA_form_ZhongShu(stock_high, start_idx, end_idx):
    # Zhongshu formed only if we found a kbar with range cover both cross
    # we only check k bars between two crosses
    s_idx = abs(start_idx)
    e_idx = abs(end_idx)
    first_cross_price = (stock_high['ma_long'][s_idx] + stock_high['ma_long'][s_idx+1]) / 2
    second_cross_price = (stock_high['ma_long'][e_idx] + stock_high['ma_long'][e_idx+1]) / 2
    
    i = s_idx+1 # start the range after first cross
    while i <= e_idx: # end the range before second cross
        if stock_high[i]['high'] > max(first_cross_price, second_cross_price) and\
            stock_high[i]['low'] < min(first_cross_price, second_cross_price):
            return True
        i = i + 1
    return False
            

def analyze_MA_zoushi_by_stock(stock,
                              period, 
                              count,
                               end_dt, 
                               df, 
                               zoushi_types, 
                               direction):
    stock_high = get_bars(stock, 
                           count=count+LONG_MA_NUM, 
                           end_dt=end_dt, 
                           unit=period,
                           fields= ['date','open',  'high', 'low','close'], 
                           df = df,
                           include_now=True)
    
    ma_long = talib.MA(stock_high['close'], LONG_MA_NUM)
    ma_short = talib.MA(stock_high['close'], SHORT_MA_NUM)
    ma_long[np.isnan(ma_long)] = 0
    ma_short[np.isnan(ma_short)] = 0
    stock_high = append_fields(stock_high,
                                ['ma_long', 'ma_short'],
                                [ma_long, ma_short],
                                [float, float],
                                usemask=False)
    
    stock_high = stock_high[LONG_MA_NUM:] # remove extra data
        
    zhongshu_results = KBar.analyze_kbar_MA_zoushi(stock_high)
    
    return KBar.analyze_kbar_MA_zoushi_exhaustion(stock_high,
                                          zoushi_types=zoushi_types,
                                           direction=direction,
                                           zhongshu=zhongshu_results)
    
def analyze_MA_exhaustion(zoushi_result, first, second):
    # work out the slope return true if the slope exhausted
    # print(first)
    # print(second)
    
    fst_max_idx = np.where(first['high'] == max(first['high']))[0][0]
    fst_min_idx = np.where(first['low'] == min(first['low']))[0][-1]
    snd_max_idx = np.where(second['high'] == max(second['high']))[0][0]
    snd_min_idx = np.where(second['low'] == min(second['low']))[0][-1]
    
    if int(fst_min_idx) == int(fst_max_idx) or\
        int(snd_min_idx) == int(snd_max_idx):
        return False
    
    if zoushi_result == ZouShi_Type.Qu_Shi_Down:
        first_slope = (max(first['high']) - min(first['low'])) / (fst_min_idx-fst_max_idx) 
        second_slope = (max(second['high']) - min(second['low'])) / (snd_min_idx-snd_max_idx) 
        return abs(second_slope) < abs(first_slope)
    elif zoushi_result == ZouShi_Type.Qu_Shi_Up:
        first=first[int(first['low'].argmin()):] # cut the data
        second=second[int(second['low'].argmin()):]
        first_slope = (max(first['high']) - min(first['low'])) / (fst_max_idx - fst_min_idx) 
        second_slope = (max(second['high']) - min(second['low'])) / (snd_max_idx - snd_min_idx) 
        return abs(second_slope) < abs(first_slope)
        
    elif zoushi_result == ZouShi_Type.Pan_Zheng:
        first_slope = (max(first['high']) - min(first['low'])) / (fst_min_idx-fst_max_idx) 
        second_slope = (max(second['high']) - min(second['low'])) / (snd_min_idx-snd_max_idx) 
        return abs(second_slope) < abs(first_slope)
    else:
        return False

def zhongshu_qushi_qualified(stock_high, direction, zhongshu):
    '''
    input zhongshu must have length of 2 or above
    make sure we have independent zhongshu along the QuShi direction
    check every zhongshu
    '''
    # check special case of initial zhongshu formed by a higher level zhongshu
    # we take the last two cross indices of the complex zhongshu
    # if len(zhongshu[0]) > 2:
    #     if direction == TopBotType.top2bot and zhongshu[0][-2] > 0:
    #         zhongshu[0] = zhongshu[0][-2:]
    #     elif direction == TopBotType.bot2top and zhongshu[0][-2] < 0:
    #         zhongshu[0] = zhongshu[0][-2:]
    # print(zhongshu)
    
    first_idx = 0
    while first_idx < len(zhongshu) - 1:
        first_zs = zhongshu[first_idx]
        second_idx = first_idx + 1
        while second_idx < len(zhongshu):
            second_zs = zhongshu[second_idx]
            if direction == TopBotType.top2bot and first_zs[0] > 0 and second_zs[0] > 0:
                first_zs_range = [(stock_high['ma_long'][first_zs[0]] + stock_high['ma_long'][first_zs[0]+1]) / 2,
                                  (stock_high['ma_long'][-first_zs[1]] + stock_high['ma_long'][-first_zs[1]+1]) / 2]
                second_zs_range = [(stock_high['ma_long'][second_zs[0]] + stock_high['ma_long'][second_zs[0]+1]) / 2,
                                   (stock_high['ma_long'][-second_zs[1]] + stock_high['ma_long'][-second_zs[1]+1]) / 2]
                if not (min(first_zs_range) > max(second_zs_range) or max(first_zs_range) < min(second_zs_range)):
                    return False
                
            elif direction == TopBotType.bot2top and first_zs[0] < 0 and second_zs[0] < 0:
                first_zs_range = [(stock_high['ma_long'][-first_zs[0]] + stock_high['ma_long'][-first_zs[0]+1]) / 2,
                                  (stock_high['ma_long'][first_zs[1]] + stock_high['ma_long'][first_zs[1]+1]) / 2]
                second_zs_range = [(stock_high['ma_long'][-second_zs[0]] + stock_high['ma_long'][-second_zs[0]+1]) / 2,
                                   (stock_high['ma_long'][second_zs[1]] + stock_high['ma_long'][second_zs[1]+1]) / 2]
                if not (min(first_zs_range) > max(second_zs_range) or max(first_zs_range) < min(second_zs_range)):
                    return False
            else:
                return False
            second_idx += 1
        first_idx += 1
    return True

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
    def analyze_kbar_MA_zoushi(cls, stock_high):
        '''
        We expect input stock data to be only one ZouShi
        '''
        # find all gold/death MA cross
        ma_diff = stock_high['ma_short'] - stock_high['ma_long']
        
        
        i = 0 
        ma_cross = [] # store the starting index of the cross + for gold - for death
        for i in range(len(ma_diff)-1):
            if ma_diff[i] < 0 and ma_diff[i+1] > 0: # gold
                ma_cross.append(i)
            elif ma_diff[i] > 0 and ma_diff[i+1] < 0: # death
                ma_cross.append(-i)
        
        
        # find all ZhongShu 
        zhongshu = []
        current_zs = []
        i = current_idx = 0
        while i < len(ma_cross) - 1:
            current_idx = i + 1
            while current_idx < len(ma_cross):
                if current_zs:
                    if analyze_MA_form_ZhongShu(stock_high, 
                                                ma_cross[i], 
                                                ma_cross[current_idx]):
                        current_zs.append(ma_cross[current_idx])
                        current_idx = current_idx + 1
                    else:
                        break
                    
                else:
                    if analyze_MA_form_ZhongShu(stock_high, 
                                                ma_cross[i], 
                                                ma_cross[current_idx]):
                        current_zs.append(ma_cross[i])
                        current_zs.append(ma_cross[current_idx])
                        current_idx = current_idx + 1
                    else:
                        break
            
            if current_zs:
                zhongshu.append(current_zs)
                current_zs = []
                    
            i = current_idx
            
        # determine ZouShi
        return zhongshu
        
    
    @classmethod
    def analyze_kbar_MA_zoushi_exhaustion(cls, stock_high, zoushi_types, direction, zhongshu):
        # gold cross -> downwards zhongshu 
        # death cross -> upwards zhongshu
        zoushi_result = ZouShi_Type.INVALID
        exhaustion_result = Chan_Type.INVALID
        
        if len(zhongshu) > 1:
            if direction == TopBotType.top2bot and\
                zhongshu_qushi_qualified(stock_high, direction, zhongshu):
                zoushi_result = ZouShi_Type.Qu_Shi_Down
            elif direction == TopBotType.bot2top and\
                zhongshu_qushi_qualified(stock_high, direction, zhongshu):
                zoushi_result = ZouShi_Type.Qu_Shi_Up
            else:
                zoushi_result = ZouShi_Type.Pan_Zheng_Composite
        elif len(zhongshu) == 1:
            zoushi_result = ZouShi_Type.Pan_Zheng
        else:
            return zoushi_result, exhaustion_result
            
        
        if zoushi_result not in zoushi_types:
            return zoushi_result, exhaustion_result
        
        # only check exhaustion by last ZhongShu
        if len(zhongshu) == 1:
            zs = zhongshu[0]
            first_part = stock_high[:abs(zs[0])+1]
            second_part = stock_high[abs(zs[-1])+1:]
            if analyze_MA_exhaustion(zoushi_result, first_part, second_part):
                exhaustion_result = Chan_Type.PANBEI
        else:
            zs1 = zhongshu[-2]
            zs2 = zhongshu[-1]
            first_part = stock_high[abs(zs1[-1])+1:abs(zs2[0])+1]
            second_part = stock_high[abs(zs2[-1])+1:]
            if analyze_MA_exhaustion(zoushi_result, first_part, second_part):
                exhaustion_result = Chan_Type.BEICHI
        
        return zoushi_result, exhaustion_result

    @classmethod
    def filter_high_level_kbar(cls, high_df, direction=TopBotType.top2bot, df=False, chan_types=[Chan_Type.III]):
        '''
        This method used by weekly (5d) data to find out rough 5m Zhongshu and 
        type III trade point
        
        It is used as initial filters for all stocks on markets
        
        1. find nearest 5m zhongshu => three consecutive weekly kbar share price region
        2. the kbar following the zhongshu escapes by direction, 
        3. the same kbar or next kbar's return attempt never touch the previous zhongshu (only strong case)
        '''
        chan_type_result = []
        num_of_kbar = TYPE_I_NUM if Chan_Type.I in chan_types else\
                    TYPE_III_NUM if Chan_Type.III in chan_types else\
                    TYPE_III_STRONG_NUM
        if df:
            if high_df.shape[0] >= num_of_kbar:
                i = -num_of_kbar
                kbar_list = []
                while i <= -1:
                    item = high_df.iloc[i]
                    kbar_list.append(item)
                    i = i + 1
            else:
                return chan_type_result
        else:
            if len(high_df['open']) >= num_of_kbar:
                kbar_list = cls.create_N_kbar(high_df, n=num_of_kbar)
            else:
                return chan_type_result
                
        if Chan_Type.III in chan_types and cls.chan_type_III_check(kbar_list, direction):
            chan_type_result.append(Chan_Type.III)
        elif Chan_Type.III_strong in chan_types and cls.chan_type_III_strong_check(kbar_list, direction, high_df):
            chan_type_result.append(Chan_Type.III_strong)
        elif (Chan_Type.I in chan_types or Chan_Type.I_weak in chan_types) and cls.chan_type_I_check(kbar_list, direction):
            chan_type_result.append(Chan_Type.I)
        elif Chan_Type.INVALID in chan_types and cls.chan_type_PB_check(kbar_list, direction):
            chan_type_result.append(Chan_Type.INVALID)
            
        return chan_type_result


    @classmethod
    def contain_zhongshu(cls, first, second, third, return_core_range=False):
        '''
        check consecutive three kbars with intersection, 
        return_core_range controls return range type, 
        in case of true only core range
        otherwise, we take the max range of first three kbars
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
                    return True, min(new_high, third.high), max(new_low, third.low)
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
    def chan_type_PB_check(cls, kbar_list, direction):
        '''
        maximum 5 Kbars, at high level used for pan bei
        '''
        if (
                direction == TopBotType.top2bot and\
                (
                    kbar_list[-5].close < kbar_list[-1].close or\
                    min(kbar_list[-2].high, kbar_list[-3].high, kbar_list[-4].high) < kbar_list[-1].close
#                     max(kbar_list[-2].low, kbar_list[-3].low, kbar_list[-4].low) > kbar_list[-5].high
                )
            ) or\
            (
                direction == TopBotType.bot2top and\
                (
                    kbar_list[-5].close > kbar_list[-1].close or\
                    max(kbar_list[-2].low, kbar_list[-3].low, kbar_list[-4].low) > kbar_list[-1].close
#                     min(kbar_list[-2].high, kbar_list[-3].high, kbar_list[-4].high) < kbar_list[-5].low
                )
            ):
            return False
        
        start_idx = -6
        end_idx = -4
        
        aa = kbar_list[start_idx]
        bb = kbar_list[start_idx+1] 
        cc = kbar_list[start_idx+2]
        last = kbar_list[-1]
        result = False
        if (last.close < last.open) or ((last.close-last.low)/(last.high-last.low) <= (1-GOLDEN_RATIO)):
            
            while start_idx <= end_idx:
                check_result, k_m, k_l = cls.contain_zhongshu(aa, bb, cc, return_core_range=False)
                
                if check_result:
                    first_max = max([kbar.high for kbar in kbar_list[:start_idx]])
                    first_min = min([kbar.low for kbar in kbar_list[:start_idx]])
                    result = (float_less_equal(last.low, k_l) and float_more_equal(first_max, k_m))\
                              if direction == TopBotType.top2bot else\
                              (float_more_equal(last.high, k_m) and float_less_equal(first_min, k_l))
                if result:
                    return result
                else:
                    start_idx = start_idx + 1
                    aa = kbar_list[start_idx]
                    bb = kbar_list[start_idx+1] 
                    cc = kbar_list[start_idx+2]
        
        return result
    
    @classmethod
    def chan_type_III_strong_check(cls, kbar_list, direction, high_df):
        '''
        We only use this check at high level 1w or 1d. with 34 K-bars. We look for simple pattern
        '''
        if direction == TopBotType.top2bot:
            high_max_loc = high_df['high'].argmax()
            if high_max_loc == 0 or high_max_loc == len(high_df['high'])-1:
                return False
            first_min_loc = high_df['low'][:high_max_loc].argmin()
            second_min_loc = high_df['low'][high_max_loc:].argmin() + high_max_loc
            
            result = second_min_loc > high_max_loc > first_min_loc and\
                     high_max_loc/ (second_min_loc + first_min_loc) > 0.382 and\
                     kbar_list[high_max_loc].high > kbar_list[second_min_loc].low > kbar_list[first_min_loc].low
        
        elif direction == TopBotType.bot2top:
            high_min_loc = high_df['low'].argmin()
            if high_min_loc == 0 or high_min_loc == len(high_df['low'])-1:
                return False
            first_max_loc = high_df['high'].argmax()
            second_max_loc = high_df['high'].argmax() + high_min_loc
            result = second_max_loc > high_min_loc > first_max_loc and\
                    high_min_loc / (second_max_loc + first_max_loc) > 0.382 and\
                    kbar_list[high_min_loc].low < kbar_list[second_min_loc].low < kbar_list[first_max_loc].low
        
        else:
        result = False
        return result
    
    @classmethod
    def chan_type_III_check(cls, kbar_list, direction):
        '''
        incase of 7 high level KBars
        assume we are given the data necessary, we need at least the last KBar to step back,
        so that means we have 4 steps for remaining 6 kbars to check zhongshu of 3 kbars
        '''
        # early check
        if (direction == TopBotType.top2bot and\
                (
                    max(kbar_list[0].high, kbar_list[1].high, kbar_list[2].high) >= kbar_list[-1].low or\
                    kbar_list[-1].close >= max(kbar_list[-2].high, kbar_list[-3].high)
                )
            ) or\
            (direction == TopBotType.bot2top and\
                (
                    min(kbar_list[0].low, kbar_list[1].low, kbar_list[2].low) <= kbar_list[-1].low or\
                    kbar_list[-1].close <= min(kbar_list[-2].low, kbar_list[-3].low)
                )
            ):
            return False
        
        start_idx = 0
        steps = 4
        first = kbar_list[start_idx]
        second = kbar_list[start_idx+1] 
        third = kbar_list[start_idx+2]
        last = kbar_list[-1]
        result = False
        if (last.close < last.open) or ((last.close-last.low)/(last.high-last.low) <= (1-GOLDEN_RATIO)):
            
            while start_idx < steps:
                check_result, k_m, k_l = cls.contain_zhongshu(first, second, third, return_core_range=False)
                
                if check_result:
                    result = (last.low > k_m if direction == TopBotType.top2bot else last.high < k_l) \
                                if (last.close < last.open) else \
                            (last.close > k_m if direction == TopBotType.top2bot else last.close < k_l)
                if result:
                    return result
                else:
                    start_idx = start_idx + 1
                    first = kbar_list[start_idx]
                    second = kbar_list[start_idx+1] 
                    third = kbar_list[start_idx+2]
        return result
    
    @classmethod
    def chan_type_I_check(cls, kbar_list, direction):
        '''
        Max number of high level Kbars.
        '''
        # early check for Type I, we expect straight up or down
        if (direction == TopBotType.top2bot and\
            (kbar_list[0].high <= kbar_list[-1].low or\
            min([kl.low for kl in kbar_list]) != kbar_list[-1].low)) or\
            (direction == TopBotType.bot2top and\
            (kbar_list[0].low >= kbar_list[-1].high or\
             max([kl.high for kl in kbar_list]) != kbar_list[-1].high)):
            return False
        
        result = False
        i = 0
        first_zs = False
        first_zs_ma = 0
        first_zs_mi = 0
        extra_count = 2 # the round of checks after finding the first ZS
        while i+4 < len(kbar_list) and extra_count > 0:
            first = kbar_list[i]
            second = kbar_list[i+1]
            third = kbar_list[i+2]
            check_result, ma, mi = cls.contain_zhongshu(first, second, third, return_core_range=False)
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
                    forth = kbar_list[i+3]
                    result = first_zs_ma < mi if direction == TopBotType.bot2top else first_zs_mi > ma and\
                            forth.close < mi if direction == TopBotType.top2bot else forth.close > ma 
                    if result:
                        break
                i = i + 1
                extra_count = extra_count - 1
        return result

        

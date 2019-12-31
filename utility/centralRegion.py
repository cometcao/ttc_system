# -*- encoding: utf8 -*-
'''
Created on 23 Dec 2019

@author: MetalInvest
'''

import numpy as np
import pandas as pd
import talib
from enum import Enum 
# from collections import OrderedDict
from utility.biaoLiStatus import * 
from utility.kBarProcessor import *

class ZhongShuLevel(Enum):
    previousprevious = -2
    previous = -1
    current = 0
    next = 1
    nextnext = 2

class Chan_Node(object):
    def __init__(self, df_node):
        self.time = df_node.name
        self.chan_price = df_node.chan_price
        self.loc = df_node.new_index
        
    def __repr__(self):
        return "price: {0} time: {1} loc: {2} ".format(self.chan_price, self.time, self.loc)
    
    def __eq__(self, node):
        return self.time == node.time and self.chan_price == node.chan_price and self.loc == node.loc

class XianDuan_Node(Chan_Node):
    def __init__(self, df_node):
        Chan_Node.__init__(self, df_node)
        self.tb = df_node.xd_tb
        self.macd_acc = df_node.macd_acc_xd_tb
        
    def __repr__(self):
        return super().__repr__() + "tb: {0}".format(self.tb)
    
    def __eq__(self, node):
        return super().__eq__(node) and self.tb == node.tb
        
class BI_Node(Chan_Node):
    def __init__(self, df_node):
        Chan_Node.__init__(self, df_node)
        self.tb = df_node.tb
        self.macd_acc = df_node.macd_acc_tb

    def __repr__(self):
        return super().__repr__() + "tb: {0}".format(self.tb)
    
    def __eq__(self, node):
        return super().__eq__(node) and self.tb == node.tb

class Double_Nodes(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        assert type(self.start) is Chan_Node, "Invalid starting node type"
        assert type(self.end) is Chan_Node, "Invalid ending node type"
        assert (start.tb == TopBotType.top and end.tb == TopBotType.bot) or (start.tb == TopBotType.bot and end.tb == TopBotType.top), "Invalid tb info" 
        assert (start.time < end.time), "Invalid node timing order"
        self.direction = TopBotType.bot2top if self.start.chan_price < self.end.chan_price else TopBotType.top2bot        

    def get_time_region(self):
        return self.start.time, self.end.time

    def work_out_slope(self):
        return (end.chan_price - start.chan_price) / (end.loc - start.loc)

class XianDuan(Double_Nodes):
    '''
    This class takes two xd nodes 
    '''
    def __init__(self, start, end):
        Double_Nodes.__init__(self, start, end)
    
    
class BI(Double_Nodes):
    '''
    This class takes two bi nodes 
    '''
    def __init__(self, start, end):
        Double_Nodes.__init__(self, start, end)
        

        
class ZouShi(object):
    '''
    This class contain the full dissasemble of current zou shi, alternative of zslx and zs
    '''
    def __init__(self, all_nodes, isdebug=False):
        self.zslx_all_nodes = all_nodes
        self.zslx_result = []
        self.isdebug = isdebug
            
    
    def analyze(self, initial_direction):
        i = 0
        temp_zslx = ZouShiLeiXing(initial_direction, [])
        previous_node = None
        while i < len(self.zslx_all_nodes) - 1:
            first = self.zslx_all_nodes[i]
            second = self.zslx_all_nodes[i+1]

            third = self.zslx_all_nodes[i+2] if i+2 < len(self.zslx_all_nodes) else None
            forth = self.zslx_all_nodes[i+3] if i+3 < len(self.zslx_all_nodes) else None
            
            if type(temp_zslx) is ZouShiLeiXing:
                if third is not None and forth is not None and ZouShiLeiXing.is_valid_central_region(temp_zslx.direction, first, second, third, forth):
                    # new zs found end previous zslx
                    if not temp_zslx.isEmpty():
                        temp_zslx.add_new_nodes(first)
                        self.zslx_result.append(temp_zslx)
                    # use previous zslx direction for new sz direction
                    temp_zslx = ZhongShu(first, second, third, forth, temp_zslx.direction)
                    if self.isdebug:
                        print("start new Zhong Shu, end previous zslx")
                    previous_node = forth
                    i = i + 2 # use to be 3, but we accept the case where last XD of ZhongShu can be zslx
                else:
                    # continue in zslx
                    temp_zslx.add_new_nodes(first)
                    if self.isdebug:
                        print("continue current zou shi lei xing: {0}".format(temp_zslx))
                    previous_node = first
                    i = i + 1
            else:
                if type(temp_zslx) is ZhongShu: 
                    ed = temp_zslx.out_of_zhongshu(first, second)
                    if ed != TopBotType.noTopBot:
                        # new zsxl going out of zs
                        self.zslx_result.append(temp_zslx)
                        temp_zslx = ZouShiLeiXing(ed, [previous_node])
                        if self.isdebug:
                            print("start new zou shi lei xing, end previous zhong shu")
                    else:
                        # continue in the zs
                        if first != temp_zslx.first and first != temp_zslx.second and first != temp_zslx.third and first != temp_zslx.forth:
                            temp_zslx.add_new_nodes(first)
                        if self.isdebug:
                            print("continue zhong shu: {0}".format(temp_zslx))
                        previous_node = first
                        i = i + 1
        
#         # add remaining nodes
        temp_zslx.add_new_nodes(self.zslx_all_nodes[i:])
        self.zslx_result.append(temp_zslx)

        return self.zslx_result

class ZouShiLeiXing(object):
    '''
    ZouShiLeiXing base class, it contain a list of nodes which represents the zou shi lei xing. 
    A central region
    B normal zou shi
    '''
    def __init__(self, direction, nodes=None):
        self.zoushi_nodes = nodes
        self.direction = direction
        
        self.amplitude_region = []
        self.time_region = []
    
    def isEmpty(self):
        return not bool(self.zoushi_nodes)
    
    def add_new_nodes(self, tb_nodes):
        if type(tb_nodes) is list:
#             list(OrderedDict.fromkeys(self.zoushi_nodes + tb_nodes))
            self.zoushi_nodes = self.zoushi_nodes + tb_nodes
        else:
            if tb_nodes not in self.zoushi_nodes:
                self.zoushi_nodes.append(tb_nodes)
        
        self.get_amplitude_region(self.zoushi_nodes)
    
    def __repr__(self):
        if self.isEmpty():
            return "Empty Zou Shi Lei Xing!"
        [s, e] = self.get_time_region()
        return "\nZou Shi Lei Xing: {0} {1}->{2}\n[\n".format(self.direction, s, e) + '\n'.join([node.__repr__() for node in self.zoushi_nodes]) + '\n]'

    @classmethod
    def is_valid_central_region(cls, direction, first, second, third, forth):
        valid = False
        if direction == TopBotType.top2bot:
            valid = first.chan_price < second.chan_price and second.chan_price > third.chan_price and third.chan_price < forth.chan_price and first.chan_price <= forth.chan_price
        elif direction == TopBotType.bot2top:
            valid = first.chan_price > second.chan_price and second.chan_price < third.chan_price and third.chan_price > forth.chan_price and first.chan_price >= forth.chan_price           
        else:
            print("Invalid direction: {0}".format(direction))
        return valid

    def get_amplitude_region(self, xd_tb_nodes=None):
        if not self.amplitude_region:
            chan_price_list = [node.chan_price for node in self.zoushi_nodes]
            self.amplitude_region = [min(chan_price_list), max(chan_price_list)]
        else:
            if xd_tb_nodes is not None:
                xd_prices = [xp.chan_price for xp in xd_tb_nodes]
                max_p = max(xd_prices)
                min_p = min(xd_prices)
                self.amplitude_region = [min(self.amplitude_region[0], min_p), max(self.amplitude_region[1], max_p)]
            else:
                chan_price_list = [node.chan_price for node in self.zoushi_nodes]
                self.amplitude_region = [min(chan_price_list), max(chan_price_list)]
        return self.amplitude_region

    def get_time_region(self):    
        if self.isEmpty():
            return [None, None]
        if not self.time_region: # assume node stored in time order
            self.time_region = [self.zoushi_nodes[0].time, self.zoushi_nodes[-1].time]
        else:
            self.zoushi_nodes.sort(key=lambda x: x.time)
            self.time_region = [self.zoushi_nodes[0].time, self.zoushi_nodes[-1].time]
        return self.time_region
    
    def get_amplitude_loc(self):
        all_node_price = [node.chan_price for node in self.zoushi_nodes]
        min_price_loc = self.zoushi_nodes[all_node_price.index(min(all_node_price))].loc
        max_price_loc = self.zoushi_nodes[all_node_price.index(max(all_node_price))].loc
        return min_price_loc, max_price_loc
    
    def work_out_slope(self):
        '''
        negative slope meaning price going down
        '''
        if not self.zoushi_nodes:
            print("Empty zslx")
            return 0
        min_price_loc, max_price_loc = self.get_amplitude_loc()
        off_set = max_price_loc - min_price_loc # this could be negative
        if np.isclose(off_set, 0.0):
            print("0 offset, INVALID")
            return 0
        
        [min_price, max_price] = self.get_amplitude_region()
        return (max_price - min_price) / off_set
    
    def get_macd_acc(self):
        top_nodes = [node for node in self.zoushi_nodes if node.tb == TopBotType.top]
        bot_nodes = [node for node in self.zoushi_nodes if node.tb == TopBotType.bot]
        macd_acc = 0.0
        if self.direction == TopBotType.bot2top:
            macd_acc = sum([node.macd_acc for node in top_nodes])
        elif self.direction == TopBotType.top2bot:
            macd_acc = sum([node.macd_acc for node in bot_nodes])
        else:
            print("We have invalid direction for ZhongShu")
        return macd_acc
    
    def check_exhaustion(self):
        '''
        check most recent two XD or BI at current direction on slopes
        '''
        i = 0
        all_double_nodes = []
        while i < len(self.zoushi_nodes)-1:
            current_node = self.zoushi_nodes[i]
            next_node = self.zoushi_nodes[i+1]
            dn = Double_Nodes(current_node, next_node)
            all_double_nodes.append(dn)
            i = i + 1
        
        same_direction_nodes = [n for n in all_double_nodes if n.direction == self.direction]
        if (self.direction == TopBotType.top2bot and same_direction_nodes[-1].end.tb == TopBotType.bot) or\
            (self.direction == TopBotType.bot2top and same_direction_nodes[-1].end.tb == TopBotType.top):
            return abs(same_direction_nodes[-1].work_out_slope()) < abs(same_direction_nodes[-3].work_out_slope())
        else:
            return False
        

class ZhongShu(ZouShiLeiXing):
    '''
    This class store all information of a central region. core data is in the format of pandas series, obtained from iloc
    The first four nodes must be in time order
    '''
    
    def __init__(self, first, second, third, forth, direction):
        ZouShiLeiXing.__init__(self, direction, [])
        self.first = first
        self.second = second
        self.third = third
        self.forth = forth
        self.extra_nodes = []

        self.core_region = []
        self.core_time_region = []
        self.core_amplitude_region = []

        self.get_core_region()
        self.get_core_amplitude_region()
        self.get_core_time_region()
        self.get_amplitude_region()
        self.get_time_region()
    
    def __repr__(self):
        [s, e] = self.get_time_region()
        return "\nZhong Shu {0}:{1}-{2}-{3}-{4} {5}->{6} level@{7}\n[".format(self.direction, self.first.chan_price, self.second.chan_price, self.third.chan_price, self.forth.chan_price, s, e, self.get_level()) + '\n'.join([node.__repr__() for node in self.extra_nodes]) + ']'        
    
    def add_new_nodes(self, tb_nodes):
        if type(tb_nodes) is list:
#             list(OrderedDict.fromkeys(self.extra_nodes + tb_nodes))
            self.extra_nodes = self.extra_nodes + tb_nodes
            self.get_amplitude_region(tb_nodes)
        else:
            self.extra_nodes.append(tb_nodes)
            self.get_amplitude_region([tb_nodes])    
    
    def out_of_zhongshu(self, node1, node2):
        [l,h] = self.get_core_region()
        exit_direction = TopBotType.noTopBot
        if (node1.chan_price < l and node2.chan_price < l):
            exit_direction = TopBotType.top2bot  
        elif (node1.chan_price > h and node2.chan_price > h):
            exit_direction = TopBotType.bot2top
        else:
            exit_direction = TopBotType.noTopBot
        return exit_direction
    
    def not_in_core(self, node):
        [l,h] = self.get_core_region()
        return node.chan_price < l or node.chan_price > h
    
    def get_core_region(self):
        upper = 0.0
        lower = 0.0
        if self.direction == TopBotType.bot2top and self.first.tb == self.third.tb == TopBotType.top and self.second.tb == self.forth.tb == TopBotType.bot:
            upper = min(self.first.chan_price, self.third.chan_price)
            lower = max(self.second.chan_price, self.forth.chan_price)
        elif self.direction == TopBotType.top2bot and self.first.tb == self.third.tb == TopBotType.bot and self.second.tb == self.forth.tb == TopBotType.top:
            lower = max(self.first.chan_price, self.third.chan_price)
            upper = min(self.second.chan_price, self.forth.chan_price)
        else:
            print("Invalid central region")       
        self.core_region = [lower, upper] 
        return self.core_region

    def get_core_amplitude_region(self):
        price_list = [self.first.chan_price, self.second.chan_price, self.third.chan_price, self.forth.chan_price]
        self.core_amplitude_region = [min(price_list), max(price_list)]
    
    def get_core_time_region(self):
        self.core_time_region = [self.first.time, self.forth.time]
        return self.core_time_region    
    
    def get_amplitude_region(self, xd_tb_nodes=None):
        if not self.amplitude_region:
            self.amplitude_region = [min(self.first.chan_price, self.second.chan_price, self.third.chan_price, self.forth.chan_price), max(self.first.chan_price, self.second.chan_price, self.third.chan_price, self.forth.chan_price)]
        else:
            if xd_tb_nodes is not None:
                xd_prices = [xp.chan_price for xp in xd_tb_nodes]
                max_p = max(xd_prices)
                min_p = min(xd_prices)
                self.amplitude_region = [min(self.amplitude_region[0], min_p), max(self.amplitude_region[1], max_p)]
            else:
                all_nodes = [self.first, self.second, self.third, self.forth] + self.extra_nodes
                all_nodes_price = [n.chan_price for n in all_nodes]
                self.amplitude_region = [min(all_nodes_price), max(all_nodes_price)]
        return self.amplitude_region    

    def get_time_region(self):    
        if not self.time_region: # assume node stored in time order
            if not self.extra_nodes:
                self.time_region = self.get_core_time_region()
            else:
                self.time_region = [self.core_time_region[0], max(self.core_time_region[-1], self.extra_nodes[-1].time)]
        else:
            if self.extra_nodes:
                self.extra_nodes.sort(key=lambda x: x.time)
                self.time_region = [self.core_time_region[0], max(self.core_time_region[-1], self.extra_nodes[-1].time)]
        return self.time_region

    def get_level(self):
        # 4 core nodes + 6 extra nodes => 9 xd as next level
        return ZhongShuLevel.current if len(self.extra_nodes) < 6 else ZhongShuLevel.next if 6 <= len(self.extra_nodes) < 24 else ZhongShuLevel.nextnext

    def take_last_xd_as_zslx(self):
        exiting_nodes = [self.forth] + self.extra_nodes if self.extra_nodes else []
        return ZouShiLeiXing(self.direction, exiting_nodes) 

    def is_complex_type(self):
        # if the ZhongShu contain more than 3 XD, it's a complex ZhongShu, in practice the direction of it can be interpreted differently
        return bool(self.extra_nodes)

    def is_running_type(self):
        running_type = False
        if self.direction == TopBotType.bot2top:
            pass
        elif self.direction == TopBotType.top2bot:
            pass
        else:
            pass
    
class CentralRegionProcess(object):
    '''
    This lib takes XD data, and the dataframe must contain chan_price, new_index, xd_tb, macd columns
    '''
    def __init__(self, kDf, high_df=None, isdebug=False, use_xd=True):    
        self.original_xd_df = kDf
        self.high_level_df = high_df
        self.use_xd = use_xd
        self.analytic_result = []
        self.isdebug = isdebug
        
        
    
    def find_initial_direction_highlevel(self):
        # higher level df data, find nearest top or bot
        # use 30m or 1d
        max_price_idx = self.high_level_df['close'].idxmax()
        min_price_idx = self.high_level_df['close'].idxmin()
        initial_idx = max(max_price_idx, min_price_idx)
        initial_direction = TopBotType.top2bot if max_price_idx > min_price_idx else TopBotType.bot2top
        if self.isdebug:
            print("initial direction: {0}, start idx {1}".format(initial_direction, initial_idx))
        return initial_idx, initial_direction  
    

    def work_out_direction(self, first, second, third, forth):
        assert first.tb == third.tb and second.tb == forth.tb, "Invalid tb information for direction"
        result_direction = TopBotType.noTopBot
        if first.tb == TopBotType.top and second.tb == TopBotType.bot:
            result_direction = TopBotType.bot2top if third.chan_price > first.chan_price else TopBotType.top2bot
        elif first.tb == TopBotType.bot and second.tb == TopBotType.top:
            result_direction = TopBotType.bot2top if third.chan_price > first.chan_price else TopBotType.top2bot
        else:
            print("Invalid tb data!!")
            
        return result_direction
    
    
    def find_initial_direction(self, working_df): 
        i = 0
        first = working_df.iloc[i]
        second = working_df.iloc[i+1]
        third = working_df.iloc[i+2]
        forth = working_df.iloc[i+3]
        
#         if ZouShiLeiXing.is_valid_central_region(TopBotType.bot2top, first, second, third, forth):
#             initial_direction = TopBotType.bot2top
#             initial_idx = working_df.index[i]
#         elif ZouShiLeiXing.is_valid_central_region(TopBotType.top2bot, first, second, third, forth):
#             initial_direction = TopBotType.bot2top
#             initial_idx = working_df.index[i]
#         else: # case of ZSLX
        initial_direction = self.work_out_direction(first, second, third, forth)
        initial_idx = working_df.index[i]
        
        if self.isdebug:
            print("initial direction: {0}, start idx {1}".format(initial_direction, initial_idx))
        return initial_idx, initial_direction  
        
    
    def find_central_region(self, initial_idx, initial_direction, working_df):
        working_df = working_df.loc[initial_idx:,:]
        
        zoushi = ZouShi([XianDuan_Node(working_df.iloc[i]) for i in range(working_df.shape[0])], isdebug=self.isdebug) if self.use_xd else ZouShi([BI_Node(working_df.iloc[i]) for i in range(working_df.shape[0])], isdebug=self.isdebug)
        return zoushi.analyze(initial_direction)
    
    def define_central_region(self):
        '''
        We probably need fully integrated stock df with xd_tb
        '''
        working_df = self.original_xd_df        
        
        working_df = self.prepare_df_data(working_df)
        
        init_idx, init_d = self.find_initial_direction(working_df)
        
        self.analytic_result = self.find_central_region(init_idx, init_d, working_df)
        
        if self.isdebug:
            print("Zou Shi disassembled: {0}".format(self.analytic_result))
            
        return self.analytic_result
        
    def convert_to_graph_data(self):
        '''
        We are assuming the Zou Shi is disassembled properly with data in timely order
        '''
        x_axis = []
        y_axis = []
        for zs in self.analytic_result:
            if type(zs) is ZhongShu:
                print(zs)
                x_axis = x_axis + zs.get_core_time_region()
                y_axis = y_axis + zs.get_core_region()
            else:
                continue
        
        return x_axis, y_axis
        
        
    def prepare_df_data(self, working_df):        
        _, _, working_df.loc[:,'macd'] = talib.MACD(working_df['close'].values)

        tb_name = 'xd_tb' if self.use_xd else 'tb'
        working_df = self.prepare_macd(working_df, tb_name)

        working_df = working_df[(working_df[tb_name]==TopBotType.top) | (working_df[tb_name]==TopBotType.bot)]
        
        if self.isdebug:
            print("working_df: {0}".format(working_df.head(10)[['chan_price', tb_name, 'new_index','macd_acc_'+tb_name]]))
        return working_df
    
    def prepare_macd(self, working_df, tb_col):
        working_df['tb_pivot'] = working_df.apply(lambda row: 0 if pd.isnull(row[tb_col]) else 1, axis=1)
        groups = working_df['tb_pivot'][::-1].cumsum()[::-1]
        working_df['tb_pivot_acc'] = groups
        
        df_macd_acc = working_df.groupby(groups)['macd'].agg([('macd_acc_negative' , lambda x : x[x < 0].sum()) , ('macd_acc_positive' , lambda x : x[x > 0].sum())])
        working_df = pd.merge(working_df, df_macd_acc, left_on='tb_pivot_acc', right_index=True)
        working_df['macd_acc_'+tb_col] = working_df.apply(lambda row: 0 if pd.isnull(row[tb_col]) else row['macd_acc_negative'] if row[tb_col] == TopBotType.bot else row['macd_acc_positive'] if row[tb_col] == TopBotType.top else 0, axis=1)
        
        working_df.drop(['tb_pivot', 'tb_pivot_acc', 'macd_acc_negative', 'macd_acc_positive'], axis=1, inplace=True)
        
        return working_df
        


# -*- encoding: utf8 -*-
'''
Created on 23 Dec 2019

@author: MetalInvest
'''

import numpy as np
import pandas as pd
import talib
from utility.biaoLiStatus import * 
from utility.kBarProcessor import *

class Chan_Node(object):
    def __init__(self, df_node):
        self.time = df_node.name
        self.chan_price = df_node.chan_price
        self.loc = df_node.new_index
        
    def __repr__(self):
        return "price: {0} time: {1} loc: {2} ".format(self.chan_price, self.time, self.loc)

class XianDuan_Node(Chan_Node):
    def __init__(self, df_node):
        Chan_Node.__init__(self, df_node)
        self.tb = df_node.xd_tb
        
    def __repr__(self):
        return super().__repr__() + "tb: {0}".format(self.tb)
        
class BI_Node(Chan_Node):
    def __init__(self, df_node):
        Chan_Node.__init__(self, df_node)
        self.tb = df_node.tb

    def __repr__(self):
        return super().__repr__() + "tb: {0}".format(self.tb)

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

class XianDuan(object):
    '''
    This class takes two xd nodes 
    '''
    def __init__(self, start, end):
        Double_Nodes.__init__(self, start, end)
    
class BI(object):
    '''
    This class takes two bi nodes 
    '''
    def __init__(self, start, end):
        Double_Nodes.__init__(self, start, end)
        
class ZouShi(object):
    '''
    This class contain the full dissasemble of current zou shi, alternative of zslx and zs
    '''
    def __init__(self, all_nodes, direction, isdebug=False):
        self.zslx_all_nodes = all_nodes
        self.direction = direction
        self.zslx_result = []
        self.isdebug = isdebug
    
    def analyze(self):
        i = 0
        temp_zslx = ZouShiLeiXing()
        previous_node = None
        while i < len(self.zslx_all_nodes) - 4:
            first = self.zslx_all_nodes[i]
            second = self.zslx_all_nodes[i+1]
            third = self.zslx_all_nodes[i+2]
            forth = self.zslx_all_nodes[i+3]
            
            if type(temp_zslx) is ZouShiLeiXing:
                if ZouShiLeiXing.is_valid_central_region(self.direction, first, second, third, forth):
                    # new zs found end previous zslx
                    if not temp_zslx.isEmpty():
                        self.zslx_result.append(temp_zslx)
                    temp_zslx = ZhongShu(first, second, third, forth, self.direction)
                    if self.isdebug:
                        print("Zhong Shu found: {0}, end previous {1}".format(temp_zslx, self.zslx_result[-1]))
                    previous_node = forth
                    i = i + 4
                else:
                    # continue in zslx
                    temp_zslx.add_new_nodes(first)
                    if self.isdebug:
                        print("continue current zou shi lei xing: {0}".format(temp_zslx))
                    previous_node = first
                    i = i + 1
            else:
                if type(temp_zslx) is Zhongshu: 
                    if temp_zslx.out_of_zhongshu(first, second):
                        # new zsxl going out of zs
                        self.zslx_result.append(temp_zslx)
                        temp_zslx = ZouShiLeiXing([previous_node, first, second])
                        if self.isdebug:
                            print("started new zou shi lei xing: {0}, end previous zhong shu {1}".format(temp_zslx, self.zslx_result[-1]))
                        previous_node = second
                        i = i + 2
                    else:
                        # continue in the zs
                        temp_zslx.add_new_nodes(first)
                        if self.isdebug:
                            print("continue zhong shu: {0}".format(temp_zslx))
                        previous_node = first
                        i = i + 1
        return self.zslx_result

class ZouShiLeiXing(object):
    '''
    ZouShiLeiXing base class, it contain a list of nodes which represents the zou shi lei xing. 
    A central region
    B normal zou shi
    '''
    def __init__(self):
        self.zoushi_nodes = []
        
#         self.entrance_node = []
#         self.exit_node = []
        
        self.amplitude_region = []
        self.time_region = []
    
    def isEmpty(self):
        return self.zoushi_nodes == []
    
    def add_new_nodes(self, tb_nodes):
        if type(tb_nodes) is list:
            self.zoushi_nodes = self.zoushi_nodes + tb_nodes
        else:
            self.zoushi_nodes.append(tb_nodes)
        
        self.get_amplitude_region(self.zoushi_nodes)
    
    def __repr__(self):
        return '\n{' + '\n'.join([node.__repr__() for node in self.zoushi_nodes]) + '}'

    @classmethod
    def is_valid_central_region(cls, direction, first, second, third, forth):
        valid = False
        if direction == TopBotType.top2bot:
            central_region = first.chan_price < second.chan_price and second.chan_price > third.chan_price and third.chan_price < forth.chan_price
        elif direction == TopBotType.bot2top:
            central_region = first.chan_price > second.chan_price and second.chan_price < third.chan_price and third.chan_price > forth.chan_price            
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
        if not self.time_region: # assume node stored in time order
            self.time_region = [self.zoushi_nodes[0].time, self.zoushi_nodes[-1].time]
        else:
            self.zoushi_nodes.sort(key=lambda x: x.time)
            self.time_region = [self.zoushi_nodes[0].time, self.zoushi_nodes[-1].time]
        return self.time_region

class ZhongShu(ZouShiLeiXing):
    '''
    This class store all information of a central region. core data is in the format of pandas series, obtained from iloc
    The first four nodes must be in time order
    '''
    
    def __init__(self, first, second, third, forth, direction):
        ZouShiLeiXing.__init__(self)
        self.direction = direction
        self.first = first
        self.second = second
        self.third = third
        self.forth = forth
        self.extra_nodes = []

        self.core_region = []
        self.core_time_region = []

        self.get_core_region()
        self.get_amplitude_region()
        self.get_CR_core_time_region()
        self.get_CR_time_region()
    
    def add_new_nodes(self, tb_nodes):
        if type(xd_tb_nodes) is list:
            self.extra_nodes = self.extra_nodes + tb_nodes
        else:
            self.extra_nodes.append(tb_nodes)
        
        self.get_amplitude_region(tb_nodes)    
    
    def out_of_zhongshu(self, node1, node2):
        [l,h] = self.get_core_region()
        return (node1.chan_price < l and node2.chan_price < l) or (node1.chan_price > h and node2.chan_price > h)
    
    def not_in_core(self, node):
        [l,h] = self.get_core_region()
        return node.chan_price < l or node.chan_price > h
    
    def get_core_region(self):
        upper = 0.0
        lower = 0.0
        if direction == TopBotType.bot2top and first.xd_tb == third.xd_tb == TopBotType.top and second.xd_tb == forth.xd_tb == TopBotType.bot:
            upper = min(first.chan_price, third.chan_price)
            lower = max(second.chan_price, forth.chan_price)
        elif direction == TopBotType.top2bot and first.xd_tb == third.xd_tb == TopBotType.bot and second.xd_tb == forth.xd_tb == TopBotType.top:
            upper = max(first.chan_price, third.chan_price)
            lower = min(second.chan_price, forth.chan_price)
        else:
            print("Invalid central region")       
        self.core_region = [lower, upper] 
        return self.core_region
    
    def get_CR_core_time_region(self):
        self.core_time_region = [first.name, forth.name]
        return self.core_time_region

    def get_entrance_node(self):
        return first
    
    def get_exit_node(self):
        return self.extra_nodes[-1]

    def get_time_region(self):    
        if not self.time_region: # assume node stored in time order
            if not self.extra_nodes:
                self.time_region = self.get_CR_core_time_region()
            else:
                self.time_region = [self.core_region[0].name, self.extra_nodes[-1].name]
        else:
            self.extra_nodes.sort(key=lambda x: x.name)
            self.time_region = [self.core_region[0].name, self.extra_nodes[-1].name]
        return self.time_region

    def get_amplitude_region(self, xd_tb_nodes=None):
        if not self.amplitude_region:
            self.amplitude_region = [min(first.chan_price, second.chan_price, third.chan_price, forth.chan_price), max(first.chan_price, second.chan_price, third.chan_price, forth.chan_price)]
        else:
            if xd_tb_nodes is not None:
                xd_prices = [xp.chan_price for xp in xd_tb_nodes]
                max_p = max(xd_prices)
                min_p = min(xd_prices)
                self.amplitude_region = [min(self.amplitude_region[0], min_p), max(self.amplitude_region[1], max_p)]
            else:
                all_nodes = [first, second, third, forth] + self.extra_nodes
                self.amplitude_region = [min(all_nodes), max(all_nodes)]
        return self.amplitude_region
    
class CentralRegionProcess(object):
    '''
    This lib takes XD data, and the dataframe must contain chan_price, new_index, xd_tb, macd columns
    '''
    def __init__(self, kDf, high_df, isdebug=False):    
        self.original_xd_df = kDf
        self.high_level_df = high_df
        self.analytic_result = []
        self.isdebug = isdebug
        
        
    
    def find_initial_direction(self, high_level_df):
        # higher level df data, find nearest top or bot
        # use 30m or 1d
        max_price_idx = high_level_df['close'].idxmax()
        min_price_idx = high_level_df['close'].idxmin()
        initial_idx = min(max_price_idx, min_price_idx)
        initial_direction = TopBotType.top2bot if max_price_idx > min_price_idx else TopBotType.bot2top
        if self.isdebug:
            print("initial direction: {0}, start idx {1}".format(initial_direction, initial_idx))
        return initial_idx, initial_direction     
    
    def find_central_region(self, initial_idx, initial_direction, working_df):
        working_df = working_df.loc[initial_idx:,:]
#         zoushi = ZouShi(working_df.apply(lambda row: XianDuan_Node(row), axis=1).tolist(), initial_direction)
        zoushi = ZouShi([XianDuan_Node(working_df.iloc[i]) for i in range(working_df.shape[0])], initial_direction, isdebug=self.isdebug)
        return zoushi.analyze()
    
    def define_central_region(self):
        '''
        We probably need fully integrated stock df with xd_tb
        '''
        working_df = self.original_xd_df        
        
        working_df = self.prepare_df_data(working_df)
        
        init_idx, init_d = self.find_initial_direction(self.high_level_df)
        
        self.analytic_result = self.find_central_region(init_idx, init_d, working_df)
        
        print(self.analytic_result)
        
        
    def prepare_df_data(self, working_df):        
        _, _, working_df.loc[:,'macd'] = talib.MACD(working_df['close'].values)

        working_df['tb_pivot'] = working_df.apply(lambda row: 0 if pd.isnull(row['xd_tb']) else 1, axis=1)
        groups = working_df['tb_pivot'][::-1].cumsum()[::-1]
        working_df['tb_pivot_acc'] = groups
        
        df_macd_acc = working_df.groupby(groups)['macd'].agg([('macd_acc_negative' , lambda x : x[x < 0].sum()) , ('macd_acc_positive' , lambda x : x[x > 0].sum())])
        working_df = pd.merge(working_df, df_macd_acc, left_on='tb_pivot_acc', right_index=True)
        working_df['macd_acc'] = working_df.apply(lambda row: 0 if pd.isnull(row['xd_tb']) else row['macd_acc_negative'] if row['xd_tb'] == TopBotType.bot else row['macd_acc_positive'] if row['xd_tb'] == TopBotType.top else 0, axis=1)
        
        working_df = working_df[(working_df['xd_tb']==TopBotType.top) | (working_df['xd_tb']==TopBotType.bot)]
        
        if self.isdebug:
            print("working_df: {0}".format(working_df.head(10)[['chan_price', 'xd_tb', 'new_index']]))
        return working_df
    
    


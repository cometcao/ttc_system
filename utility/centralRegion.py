# -*- encoding: utf8 -*-
'''
Created on 23 Dec 2019

@author: MetalInvest
'''

import numpy as np
from utility.biaoLiStatus import * 
from utility.kBarProcessor import *

class XianDuan(object):
    '''
    This class takes two nodes 
    '''
    def __init__(self, start, end):
        self.start = start
        self.end = end
        assert (start.xd_tb == TopBotType.top and end.xd_tb == TopBotType.bot) or (start.xd_tb == TopBotType.bot and end.xd_tb == TopBotType.top), "Invalid XD xd_tb info" 
        self.direction = TopBotType.bot2top if self.start.chan_price < self.end.chan_price else TopBotType.top2bot
    
    def get_time_region(self):
        return self.start.name, self.end.name

class CentralRegion(object):
    '''
    This class store all information of a central region. core data is in the format of pandas series, obtained from iloc
    The first four nodes must be in time order
    '''
    def __init__(self, first, second, third, forth, direction):
        self.first = first
        self.second = second
        self.third = third
        self.forth = forth
        self.direction = direction
        self.core_region = []
        self.amplitude_region = []
        self.extra_XD = []
        self.core_time_region = []
        self.time_region = []
        self.entrance_XD = []
        self.exit_XD = []
        
        self.get_core_region()
        self.get_amplitude_region()
        self.get_CR_core_time_region()
        self.get_CR_time_region()
    
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
                all_nodes = [first, second, third, forth] + self.extra_XD
                self.amplitude_region = [min(all_nodes), max(all_nodes)]
        return self.amplitude_region
        
    def is_valid_central_region(self, direction, first, second, third, forth):
        valid = False
        if direction == TopBotType.top2bot:
            central_region = self.first.chan_price < self.second.chan_price and self.second.chan_price > self.third.chan_price and self.third.chan_price < self.forth.chan_price
        elif direction == TopBotType.bot2top:
            central_region = self.first.chan_price > self.second.chan_price and self.second.chan_price < self.third.chan_price and self.third.chan_price > self.forth.chan_price            
        else:
            print("Invalid direction: {0}".format(direction))
        return valid
    
    def add_new_xd(self, xd_tb_nodes):
        if type(xd_tb_nodes) is list:
            self.extra_XD = self.extra_XD + xd_tb_nodes
        else:
            self.extra_XD.append(xd_tb_nodes)
        
        self.get_amplitude_region(xd_tb_nodes)
    
    def get_CR_core_time_region(self):
        self.core_time_region = [first.name, forth.name]
        return self.core_time_region
            
    def get_CR_time_region(self):    
        if not self.time_region: # assume node stored in time order
            if not self.extra_XD:
                self.time_region = self.get_CR_core_time_region()
            else:
                self.time_region = [self.core_region[0], self.extra_XD[-1].name]
        else:
            self.extra_XD.sort(key=lambda x: x.name)
            self.time_region = [self.core_region[0], self.extra_XD[-1].name]
        return self.time_region

class ZouShiLeiXing(object):
    '''
    ZouShiLeiXing base class
    '''
    def __init__(self):
        pass


    
class CentralRegionProcess(object):
    '''
    This lib takes XD data, and the dataframe must contain chan_price, new_index, xd_tb, macd columns
    '''
    def __init__(self, kDf, isdebug=False):    
        self.original_xd_df = kDf
        
        
    
    def find_initial_direction(self, working_df):
        # use simple solution to find initial direction
        max_price_idx = working_df.['chan_price'].idxmax()
        min_price_idx = working_df.['chan_price'].idxmin()
        initial_idx = min(max_price_idx, min_price_idx)
        initial_direction = TopBotType.bot2top if max_price_idx > min_price_idx else TopBotType.top2bot
        return working_df.index.get_loc(initial_idx), initial_direction     
    
    def find_central_region(self, initial_loc, initial_direction, working_df):
        
        
    
    def define_central_region(self):
        '''
        We probably need fully integrated stock df with xd_tb
        '''
        working_df = self.original_xd_df        
        
        self.prepare_df_data(working_df)
        
        init_loc, init_d = self.find_initial_direction(working_df)
        
        self.find_central_region(init_loc, init_d, working_df)
        
        
        
        
    def prepare_df_data(self, working_df):
        working_df['tb_pivot'] = df.apply(lambda row: 0 if pd.isnull(row['xd_tb']) else 1, axis=1)
        groups = working_df['tb_pivot'][::-1].cumsum()[::-1]
        working_df['tb_pivot_acc'] = groups
         
        df_macd_acc = working_df.groupby(groups)['macd'].agg([('macd_acc_negative' , lambda x : x[x < 0].sum()) , ('macd_acc_positive' , lambda x : x[x > 0].sum())])
        working_df = pd.merge(working_df, df_macd_acc, left_on='tb_pivot_acc', right_index=True)
        working_df['macd_acc'] = working_df.apply(lambda row: 0 if pd.isnull(row['tb']) else row['macd_acc_negative'] if row['xd_tb'] == TopBotType.bot else row['macd_acc_positive'] if row['xd_tb'] == TopBotType.top else 0, axis=1)
    
    


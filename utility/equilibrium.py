from utility.biaoLiStatus import * 
from utility.kBarProcessor import *
from utility.centralRegion import *

import numpy as np
import pandas as pd
from pytz.reference import Central


class Equilibrium():
    '''
    This class use ZouShi analytic results to check BeiChi
    '''
    
    def __init__(self, df_all, analytic_result, isdebug=False):
        self.original_df = df_all
        self.analytic_result = analytic_result
        self.isdebug = isdebug
        self.isQvShi = False
        pass
    
    def find_most_recent_zoushi(self):
        '''
        Make sure we return the most recent Zhong Shu and the Zou Shi Lei Xing Entering it.
        The Zou Shi Lei Xing Exiting it will be reworked on the original df
        '''
        if type(self.analytic_result[-1]) is ZhongShu:
            filled_zslx = self._analytic_result[-1].take_last_xd_as_zslx()
            return self.analytic_result[-2], self._analytic_result[-1], filled_zslx
        elif type(self.analytic_result[-1]) is ZouShiLeiXing:
            return self.analytic_result[-3], self.analytic_result[-2], self.analytic_result[-1]
        else:
            print("Invalid Zou Shi type")
            return None, None, None
    
    def define_equilibrium(self):
        # check if current status beichi or panzhengbeichi
        self.check_equilibrium_status()
        
        a, B, c = self.find_most_recent_zoushi()
        
        self.check_exhaustion(a, B, c)
        
    def check_exhaustion(self, zslx_a, zs_B, zslx_c):
        zslx_slope = zslx_a.work_out_slope()
        
        latest_slope = zslx_c.work_out_slope()

        if np.sign(latest_slope) == np.sign(zslx_slop) and abs(latest_slope) < abs(zslx_slope):
            return True
        
        if self.isQvShi: # if QV SHI => at least two Zhong Shu, We could also use macd for help
            zslx_macd = zslx_a.get_macd_acc()
            latest_macd = zslx_c.get_macd_acc()
            return abs(zslx_macd) > abs(latest_macd)
        return False
         
    def check_equilibrium_status(self):
        recent_zoushi = self.analytic_result[-4:]
        recent_zhongshu = []
        for zs in recent_zoushi:
            if type(zs) is ZhongShu:
                recent_zhongshu.append(zs)
                
        if len(recent_zhongshu) > 2 and recent_zhongshu[-2].direction == recent_zhongshu[-1].direction:
            if recent_zhongshu[-2].get_level() == recent_zhongshu.get_level():
                [l1, u1] = recent_zhongshu[-2].get_amplitude_region()
                [l2, u2] = recent_zhongshu[-1].get_amplitude_region()
                if not (l1 > u2 or l2 > u1): # two Zhong Shu without intersection
                    self.isQvShi = True
        else:
            self.isQvShi = False
            
        return self.isQvShi
    
    
            
            
        
                
# -*- encoding: utf8 -*-
'''
Created on 09 Feb 2020

@author: MetalInvest
'''
import numpy as np
from enum import Enum 
######################## common method ###############################

def float_less(a, b):
    return a < b and not np.isclose(a, b)

def float_more(a, b):
    return a > b and not np.isclose(a, b)

def float_less_equal(a, b):
    return a < b or np.isclose(a, b)

def float_more_equal(a, b):
    return a > b or np.isclose(a, b)

def float_equal(a, b):
    return np.isclose(a, b)


######################## chan_kbar_filter ##########################
TYPE_III_NUM = 7
TYPE_I_NUM = 10
TYPE_III_STRONG_NUM = 89

######################## kBarprocessor #############################
GOLDEN_RATIO = 0.618
MIN_PRICE_UNIT=0.01

PRICE_UPPER_LIMIT = 200

######################## Central Region ###########################

class ZhongShuLevel(Enum):
    previousprevious = -2
    previous = -1
    current = 0
    next = 1
    nextnext = 2
    @classmethod
    def value2type(cls, val):
        if val == -2:
            return cls.previousprevious
        elif val == -1:
            return cls.previous
        elif val == 0:
            return cls.current
        elif val == 1:
            return cls.next
        elif val == 2:
            return cls.nextnext
        else:
            return cls.current
    
class ZouShi_Type(Enum):
    INVALID = -2
    Qu_Shi_Down = -1
    Pan_Zheng = 0
    Qu_Shi_Up = 1
    Pan_Zheng_Composite = 2
    @classmethod
    def value2type(cls,val):
        if val == 0:
            return cls.Pan_Zheng
        elif val == 1:
            return cls.Qu_Shi_Up
        elif val == -1:
            return cls.Qu_Shi_Down
        elif val == 2:
            return cls.Pan_Zheng_Composite
        elif val == -2:
            return cls.INVALID
    
class Chan_Type(Enum):
    INVALID = 0
    I = 1
    II = 2
    III = 3
    III_weak = 4
    II_weak = 5
    III_strong = 6
    I_weak = 7
    PANBEI = 8
    BEICHI = 9
    
    @classmethod
    def value2type(cls, val):
        if val == 0:
            return cls.INVALID
        elif val == 1:
            return cls.I
        elif val == 2:
            return cls.II
        elif val == 3:
            return cls.III
        elif val == 4:
            return cls.III_weak
        elif val == 5:
            return cls.II_weak
        elif val == 6:
            return cls.III_strong
        elif val == 7:
            return cls.I_weak
        elif val == 8:
            return cls.PANBEI
        elif val == 9:
            return cls.BEICHI
        else:
            return cls.INVALID
        
        
###################################################
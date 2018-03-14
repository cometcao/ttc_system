'''
Created on 28 Feb 2018

@author: MetalInvest
'''
try:
    from rqdatac import *
except:
    pass
try:
    from jqdata import *
except:
    pass

from enum import Enum 
import datetime


class DataRetriever():
    @staticmethod
    def get_data(security, start_date='2006-01-01', end_date=str(datetime.datetime.today()), count=100, period='1d', fields=None, skip_suspended=False, adjust_type='pre'):
        pass

    @staticmethod
    def get_research_data(security, start_date='2006-01-01', end_date=str(datetime.datetime.today()), count=100, period='1d', fields=None, skip_suspended=False, adjust_type='pre'):
        pass


class JqDataRetriever(DataRetriever):
    @staticmethod
    def get_data(security, start_date='2006-01-01', end_date=str(datetime.datetime.today()), count=100, period='1d', fields=None, skip_suspended=False, adjust_type='pre', df=True):
        return attribute_history(security, count, period, fields = fields, skip_paused=skip_suspended, df=df)
    @staticmethod
    def get_research_data(security, start_date='2006-01-01', end_date=str(datetime.datetime.today()), count=100, period='1d', fields=None, skip_suspended=False, adjust_type='pre', df=True):
        return get_price(security, count=count, end_date=end_date, frequency=period, fields = fields, skip_paused=skip_suspended, df=df)

class RqDataRetriever(DataRetriever):
    @staticmethod
    def get_data(security, count=100, period='1d', fields=None, skip_suspended=False, adjust_type='pre', df=True, include_now=False):
        if 'datetime' not in fields:
            fields.append('datetime')
        data_array = history_bars(security, bar_count=count, frequency=period, fields = fields, skip_suspended=skip_suspended, include_now=include_now)
        if df:
            data_array = pd.DataFrame(data_array, columns=fields)
        return data_array

    @staticmethod
    def get_research_data(security, start_date='2006-01-01', end_date=None, period='1d', fields=None, skip_suspended=False, adjust_type='pre', df=True):
        data_df = get_price(security, start_date=start_date, end_date=end_date, frequency=period, fields = fields, skip_paused=skip_suspended, adjust_type=adjust_type)
        if not df:
            data_df = data_df.values
        return data_df

class SecurityDataManager():
    """
    This class is the ultimate class to provide all sourcing data for TTC system
    """
    @classmethod
    def get_research_data_jq(cls, security, start_date='2006-01-01', end_date=str(datetime.datetime.today()), count=100, period='1d', fields=None, skip_suspended=False, adjust_type='pre', df=True):
        return JqDataRetriever.get_research_data(security, start_date, end_date, count, period, fields, skip_suspended, adjust_type, df)
    
    @classmethod
    def get_run_data_jq(cls, security, start_date='2006-01-01', end_date=str(datetime.datetime.today()), count=100, period='1d', fields=None, skip_suspended=False, adjust_type='pre', df=True):
        return JqDataRetriever.get_data(security, start_date, end_date, count, period, fields, skip_suspended, adjust_type, df)
    
    @classmethod
    def get_research_data_rq(cls, security, start_date='2006-01-01', end_date=None, period='1d', fields=None, skip_suspended=False, adjust_type='pre', df=True):
        return RqDataRetriever.get_research_data(security, start_date, end_date, period, fields, skip_suspended, adjust_type, df)
        
    @classmethod
    def get_data_rq(cls, security, count=100, period='1d', fields=None, skip_suspended=False, adjust_type='pre', df=True, include_now=False):
        return RqDataRetriever.get_data(security, count, period, fields, skip_suspended, adjust_type, df, include_now)
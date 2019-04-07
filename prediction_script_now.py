from utility.utility_ts import *
from utility.securityDataManager import *
from utility.ML_main import *
from utility.ML_kbar_prep import *

pd.options.mode.chained_assignment = None


stocks = ['300072.XSHE', '002072.XSHE']


# dates = get_trading_date_ts(count=15)
dates = JqDataRetriever.get_trading_date(count=1)
for day in dates[-1:]:
    
    mbc_week = ML_biaoli_check({'rq':False, 
                           'ts':False,
                           'model_path':'./training_model/weekly_model/cnn_lstm_model_base_weekly2.h5', 
                           'isAnal':True,
                           'extra_training':False, 
                           'extra_training_period':250, # 1250
                           'save_new_model':False,
                           'long_threthold':0.92, 
                           'short_threthold':0.92, 
                           'isDebug':True, 
                           'use_latest_pivot':False, 
                           'use_standardized_sub_df':True, 
                           'use_cnn_lstm':True,
                           'use_cnn':False,
                           'check_level':['5d','150m'],
                           'sub_level_max_length':240}) # 240
    gauge_results_week = mbc_week.gauge_stocks_analysis(stocks, today_date=day, check_status=True)
    
    mbc_day = ML_biaoli_check({'rq':False, 
                           'ts':False,
                           'model_path':'./training_model/subprocessed/cnn_lstm_model_base2.h5', 
                           'isAnal':True,
                           'extra_training':False, 
                           'extra_training_period':2500, # 1250
                           'save_new_model':False,
                           'long_threthold':0.92, 
                           'short_threthold':0.92, 
                           'isDebug':True, 
                           'use_latest_pivot':False, 
                           'use_standardized_sub_df':True,
                           'use_cnn_lstm':True,
                           'use_cnn':False,
                           'check_level':['1d','30m'],
                           'sub_level_max_length':400})
    gauge_results_day = mbc_day.gauge_stocks_analysis(stocks, today_date=day, check_status=True)
    print(gauge_results_week)
    print(gauge_results_day)


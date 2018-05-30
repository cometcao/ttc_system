from utility.utility_ts import *
from ML_main import *

pd.options.mode.chained_assignment = None
mbc = ML_biaoli_check({'rq':False, 
                       'ts':False,
                       'model_path':'training_model/subprocessed/cnn_lstm_model_base.h5', 
                       'isAnal':True,
                       'extra_training':False, 
                       'extra_training_period':250, # 1250
                       'save_new_model':False,
                       'long_threthold':0.9, 
                       'short_threthold':0.9, 
                       'isDebug':True, 
                       'use_latest_pivot':False})

dates = get_trading_date_ts(count=15)
for day in dates[-1:]:
    gauge_results = mbc.gauge_stocks_analysis(['000421.XSHE'], today_date=day.date())


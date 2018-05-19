from utility.utility_ts import *
from ML_main import *

pd.options.mode.chained_assignment = None
mbc = ML_biaoli_check({'rq':False, 
                       'ts':True,
                       'model_path':'training_model/cnn_lstm_model_base.h5', 
                       'isAnal':True,
                       'extra_training':True, 
                       'extra_training_period':250, # 1250
                       'save_new_model':True,
                       'long_threthold':0.9, 
                       'short_threthold':0.9, 
                       'isDebug':False, 
                       'use_latest_pivot':True})

dates = get_trading_date_ts(count=15)
for day in dates[-1:]:
    gauge_results = mbc.gauge_stocks_analysis(['002223.XSHE'], today_date=day)


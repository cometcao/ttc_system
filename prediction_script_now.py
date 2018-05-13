from ML_main import *
pd.options.mode.chained_assignment = None
mbc = ML_biaoli_check({'rq':False, 
                       'model_path':'training_model/cnn_lstm_model_base.h5', 
                       'isAnal':True,
                       'extra_training':False, 
                       'extra_training_period':500, # 1250
                       'save_new_model':False,
                       'long_threthold':0.9, 
                       'short_threthold':0.9, 
                       'isDebug':True, 
                       'use_latest_pivot':True})

dates = get_trade_days(count=15)
for day in dates[-1:]:
    gauge_results = mbc.gauge_stocks_analysis(['002223.XSHE'], today_date=day)


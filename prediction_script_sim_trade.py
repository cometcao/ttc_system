# run during lunch break
from ML_main import *
from pickle import dump
from pickle import load
import json
pd.options.mode.chained_assignment = None
dates = get_trade_days(count=250)
result = {}

mbc = ML_biaoli_check({'rq':False, 
                       'model_path':'training_model/cnn_lstm_model_base.h5', 
                       'isAnal':True,
                       'extra_training':False,
                       'extra_training_period':90,
                       'save_new_model':False,
                       'long_threthold':0.9, 
                       'short_threthold':0.9, 
                       'isDebug':False,
                       'use_latest_pivot':False})

with open("multi_factor_trading_picked_stocks.txt", 'r') as myfile:
    stocks=myfile.read().replace('\n', '')
    print(stocks)
    stocks = stocks.split(',')
    
stocks.sort()
dates = get_trade_days(count=1)

gauge_results = mbc.gauge_stocks_analysis(stocks, check_status=True, today_date=dates[-1])
result[str(dates[-1])]=[(stock, 1 if long else 0, 1 if short else 0) for stock, (long, short) in gauge_results]

result_json=json.dumps(result)
filename = './training_result/multi_factor_trading_picked_stocks.txt'
write_file(filename, str.encode(result_json, "utf8"))
print('Saved: %s' % filename)
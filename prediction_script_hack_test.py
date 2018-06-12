# for hack test
from ML_main import *
from pickle import dump
from pickle import load
from utility.utility_ts import *
import json

dates = get_trading_date_ts(count=30)
result = {}

mbc = ML_biaoli_check({'rq':False, 
                       'ts':False,
                       'model_path':'training_model/subprocessed/002001.XSHE.h5', 
                       'isAnal':True,
                       'extra_training':False,
                       'extra_training_period':1250,
                       'save_new_model':False,
                       'long_threthold':0.6, 
                       'short_threthold':0.6, 
                       'isDebug':True,
                       'use_latest_pivot':False})
# stocks = get_index_stocks('399300.XSHE')
stocks = ['002001.XSHE']
stocks.sort()
for pre_stock in stocks:
    for day in dates:
        gauge_results = mbc.gauge_stocks_analysis([pre_stock], today_date=day.date())
        result[str(day.date())]=[(stock, 1 if long else 0, 1 if short else 0) for stock, (long, short) in gauge_results]
        
#     print(result)
#     filename = 'training_result/{0}.pkl'.format(pre_stock)
#     dump(result, open(filename, 'wb'))

    filename = './training_result/{0}.jq'.format(pre_stock)
    with open(filename, 'w', encoding='utf-8') as outfile:
        json.dump(result, outfile)

#     result_json=json.dumps(result)
#     write_file(filename, str.encode(result_json, "utf8"))
    print('Saved: %s' % filename)
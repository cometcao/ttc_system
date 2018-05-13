# for hack test
from ML_main import *
from pickle import dump
from pickle import load
import json

dates = get_trade_days(count=250)
result = {}

mbc = ML_biaoli_check({'rq':False, 
                       'model_path':'training_model/cnn_lstm_model_index.h5', 
                       'isAnal':True,
                       'extra_training':False,
                       'extra_training_period':1250,
                       'save_new_model':False,
                       'long_threthold':0.999, 
                       'short_threthold':0.99, 
                       'isDebug':True})
# stocks = get_index_stocks('399300.XSHE')
stocks = ['000300.XSHG', '000016.XSHG', '399333.XSHE', '399006.XSHE']
stocks.sort()
for pre_stock in stocks:
    for day in dates:
        gauge_results = mbc.gauge_stocks_analysis([pre_stock], today_date=day)
        result[str(day)]=[(stock, 1 if long else 0, 1 if short else 0) for stock, (long, short) in gauge_results]
        
#     print(result)
#     filename = 'training_result/{0}.pkl'.format(pre_stock)
#     dump(result, open(filename, 'wb'))

    result_json=json.dumps(result)
    filename = 'training_result/{0}.jq'.format(pre_stock)
    write_file(filename, str.encode(result_json, "utf8"))
    print('Saved: %s' % filename)
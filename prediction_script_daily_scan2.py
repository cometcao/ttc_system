from utility.securityDataManager import *
from utility.ML_main import *

pd.options.mode.chained_assignment = None

result = {}
filename = 'multi_factor_trading_picked_stocks.txt'
try:
    with open(filename, "r") as result_file:
        result = json.load(result_file)
        print(result)
except Exception as e:
    print("{0} loading error {1}".format(filename, str(e)))

stocks = ["000016.XSHG"]

mbc_5d = ML_biaoli_check({'rq':False, 
                       'ts':False,
                       'model_path':'./training_model/subprocessed/rnn_cnn_model_base_5d30m_tuned2.h5', 
                       'isAnal':True,
                       'extra_training':False, 
                       'extra_training_period':250, # 1250
                       'save_new_model':False,
                       'long_threthold':0.97, 
                       'short_threthold':0.97, 
                       'isDebug':True, 
                       'use_latest_pivot':True, 
                       'use_standardized_sub_df':True, 
                       'use_cnn_lstm':False,
                       'use_cnn':False,
                       'check_level':['5d','30m'],
                       'monitor_fields':['chan_price', 'new_index', 'macd_acc', 'money_acc'],
                       'sub_level_max_length':269}) 

mbc_1d = ML_biaoli_check({'rq':False, 
                       'ts':False,
                       'model_path':'./training_model/subprocessed/rnn_cnn_model_base_1d5m.h5', 
                       'isAnal':True,
                       'extra_training':False, 
                       'extra_training_period':250, # 1250
                       'save_new_model':False,
                       'long_threthold':0.97, 
                       'short_threthold':0.97, 
                       'isDebug':True, 
                       'use_latest_pivot':True, 
                       'use_standardized_sub_df':True, 
                       'use_cnn_lstm':False,
                       'use_cnn':False,
                       'check_level':['1d','5m'],
                       'monitor_fields':['chan_price', 'new_index', 'macd_acc', 'money_acc'],
                       'sub_level_max_length':321}) 

dates = JqDataRetriever.get_trading_date(count=100)
for day in dates:
    if str(day) in result:
        print("{0} already done".format(str(day)))
        continue    
    
    print("{0}:{1}".format(str(day), stocks))
    result_5d = mbc_5d.gauge_stocks_analysis(stocks, today_date=day, check_status=True)
    result_1d = mbc_1d.gauge_stocks_analysis(stocks, today_date=day, check_status=True)
    
    combined_results = zip(result_5d, result_1d)
    
    result[str(day)]=[(stock, 1 if (long_week and long_day) else 0, 1 if (short_week or short_day) else 0) 
                            for (stock, (long_week, short_week)), (stock, (long_day, short_day)) in combined_results]
            
    print(result)
    result_json=json.dumps(result)
    with open(filename, "w+") as result_file:
#             result_file.write(str.encode(result_json, "utf8"))
        result_file.write(result_json)
#             json.dump(result, filename)
    print('Saved: %s' % filename)


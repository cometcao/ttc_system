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

# stocks = ["000016.XSHG", "000300.XSHG", "399437.XSHE", "000134.XSHG", "000010.XSHG"]
stocks = ["510180.XSHG", 
          "510300.XSHG", 
          "510050.XSHG", 
          "512800.XSHG", 
          "512000.XSHG", 
          "510230.XSHG", 
          "510310.XSHG", 
          "518880.XSHG"]

mbc_5d = ML_biaoli_check({'rq':False, 
                       'ts':False,
                       'model_path':'./training_model/weekly_model/4_classes/rnn_cnn_model_base_5d30m.h5', 
                       'isAnal':True,
                       'extra_training':False, 
                       'extra_training_period':250, # 1250
                       'save_new_model':False,
                       'long_threthold':0.9, 
                       'short_threthold':0.9, 
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
                       'model_path':'./training_model/subprocessed/4_classes/rnn_cnn_model_base_1d5m.h5', 
                       'isAnal':True,
                       'extra_training':False, 
                       'extra_training_period':250, # 1250
                       'save_new_model':False,
                       'long_threthold':0.9, 
                       'short_threthold':0.9, 
                       'isDebug':True, 
                       'use_latest_pivot':True, 
                       'use_standardized_sub_df':True, 
                       'use_cnn_lstm':False,
                       'use_cnn':False,
                       'check_level':['1d','5m'],
                       'monitor_fields':['chan_price', 'new_index', 'macd_acc', 'money_acc'],
                       'sub_level_max_length':321}) 

dates = JqDataRetriever.get_trading_date(count=300)
for day in dates:
    if str(day) in result:
        print("{0} already done".format(str(day)))
        continue    
    
    print("{0}:{1}".format(str(day), stocks))
    result_5d = mbc_5d.gauge_stocks_analysis_status(stocks, today_date=day)
    result_1d = mbc_1d.gauge_stocks_analysis_status(stocks, today_date=day)
    
    combined_results = zip(result_5d, result_1d)
    
    result[str(day)]=[(stock, float(week_status), float(day_status)) for (stock, week_status), (stock, day_status) in combined_results]

    print(result)
    result_json=json.dumps(result)
    with open(filename, "w+") as result_file:
#             result_file.write(str.encode(result_json, "utf8"))
        result_file.write(result_json)
#             json.dump(result, filename)
    print('Saved: %s' % filename)


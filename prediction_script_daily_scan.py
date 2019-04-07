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

stock_file_path = "C:/Users/MetalInvest/.joinquant-py3/notebook/2c71ebaa926a45429189bfff6423661b/daily_stocks"

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
                       'use_standardized_sub_df':True, #  no need for prediction
                       'use_cnn_lstm':True,
                       'use_cnn':False,
                       'check_level':['1d','30m'],
                       'sub_level_max_length':400})


dates = JqDataRetriever.get_trading_date(count=300)
for day in dates:
    if str(day) in result:
        print("{0} already done".format(str(day)))
        continue    
    
    with open("{0}/{1}.txt".format(stock_file_path,str(day)), 'r') as myfile:
        stocks=myfile.read().replace('\n', '')
        stocks = stocks.split(',')
        stocks.sort()        
        print("{0}:{1}".format(str(day), stocks))
        gauge_results_week = mbc_week.gauge_stocks_analysis(stocks, today_date=day, check_status=True)
        gauge_results_day = mbc_day.gauge_stocks_analysis(stocks, today_date=day, check_status=True)
        
        combined_results = zip(gauge_results_week, gauge_results_day)
        
        result[str(day)]=[(stock, 1 if (long_week and long_day) else 0, 1 if (short_week or short_day) else 0) 
                                for (stock, (long_week, short_week)), (stock, (long_day, short_day)) in combined_results]
        
        result_json=json.dumps(result)
        with open(filename, "w+") as result_file:
#             result_file.write(str.encode(result_json, "utf8"))
            result_file.write(result_json)
#             json.dump(result, filename)
        print('Saved: %s' % filename)


# auto run on trading day at 12:00

# run during lunch break
from ML_main import *
from pickle import dump
from pickle import load
import datetime
from datetime import datetime, time, date, timedelta
import json
pd.options.mode.chained_assignment = None
result = {}
runTime = '12:00'
marketCloseTime = '14:45'
startTime = time(*(map(int, runTime.split(':'))))
endTime = time(*(map(int, marketCloseTime.split(':'))))


def run_daily_ml(dates):
    mbc_weekly = ML_biaoli_check({'rq':False, 
                           'ts':False,
                           'model_path':'training_model/cnn_lstm_model_base_weekly2.h5', 
                           'isAnal':True,
                           'extra_training':False,
                           'extra_training_period':90,
                           'save_new_model':False,
                           'long_threthold':0.9, 
                           'short_threthold':0.9, 
                           'isDebug':False,
                           'use_latest_pivot':True, 
                           'use_standardized_sub_df':True,
                           'use_cnn_lstm':True,
                           'use_cnn':False,
                           'check_level':['5d','1d'],
                           'sub_level_max_length':240
                          })
    
    mbc_day = ML_biaoli_check({'rq':False, 
                           'ts':False,
                           'model_path':'training_model/cnn_lstm_model_base2.h5', 
                           'isAnal':True,
                           'extra_training':False, 
                           'extra_training_period':250, # 1250
                           'save_new_model':False,
                           'long_threthold':0.9, 
                           'short_threthold':0.9, 
                           'isDebug':False, 
                           'use_latest_pivot':True, 
                           'use_standardized_sub_df':True,
                           'use_cnn_lstm':True,
                           'use_cnn':False,
                           'check_level':['1d','30m'],
                           'sub_level_max_length':400})

    with open("multi_factor_trading_picked_stocks.txt", 'r') as myfile:
        stocks=myfile.read().replace('\n', '')
        print(stocks)
        stocks = stocks.split(',')

    stocks.sort()

    weekly_gauge_results = mbc_weekly.gauge_stocks_analysis(stocks, check_status=True, today_date=dates[-1])
    daily_gauge_results = mbc_day.gauge_stocks_analysis(stocks, check_status=True, today_date=dates[-1])
    
#     print(weekly_gauge_results)
#     print(daily_gauge_results)
    
    combined_results = zip(weekly_gauge_results, daily_gauge_results)
    
    result[str(dates[-1])]=[(stock, 1 if (long_week and long_day) else 0, 1 if (short_week or short_day) else 0) 
                            for (stock, (long_week, short_week)), (stock, (long_day, short_day)) in combined_results]

    result_json=json.dumps(result)
    filename = 'training_result/multi_factor_trading_picked_stocks.txt'
    write_file(filename, str.encode(result_json, "utf8"))
    print('Saved: %s' % filename)


while True:
    import time
    dates = get_trade_days(count=250)
    today = datetime.today().date()
    now = datetime.today().time()
    print("current time:{0}, starting time:{1}".format(now, startTime))
    
    if today not in dates: # today is trading day
        print("non-trading day, wait for 4 hours")
        time.sleep(14400) # sleep for 4 hours

    if now < startTime:
        td = datetime.combine(date.today(), startTime) - datetime.combine(date.today(), now)
        print("sleep till starting time for {0} seconds".format(td.total_seconds()))
        time.sleep(int(td.total_seconds()))
        
    run_daily_ml(dates)
    print("finished wait for next trading day")
    if now < endTime:
        time.sleep(86400) # sleep for one day
    else:
        time.sleep(32400) # sleep for 9 hours
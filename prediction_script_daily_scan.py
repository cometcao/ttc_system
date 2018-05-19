from ML_main import *
mbc = ML_biaoli_check({'rq':False, 
                       'ts':True,
                       'model_path':'training_model/cnn_lstm_model_base.h5', 
                       'isAnal':True,
                       'extra_training':False, 
                       'extra_training_period':250,
                       'save_new_model':False,
                       'long_threthold':0.999, 
                       'short_threthold':0.99, 
                       'isDebug':False})

with open("daily_scanned_stock_list.txt", 'r') as myfile:
    data=myfile.read().replace('\n', '')
    print(data)
    print(mbc.gauge_stocks_analysis(data.split(",")))
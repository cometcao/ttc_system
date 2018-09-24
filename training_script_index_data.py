########## Index Model #############

from utility.ML_main import *
from utility.ML_kbar_prep import *
from utility.ML_model_prep import *


pd.options.mode.chained_assignment = None

# from tensorflow.python.client import device_lib
# 
# print(device_lib.list_local_devices())

mbt = ML_biaoli_train({'ts':False,
                       'rq':False, 
                       'isAnal':True, 
                       'index_list':['000016.XSHG','000905.XSHG','399300.XSHE', '000001.XSHG', '399001.XSHE', '399333.XSHE', '399006.XSHE'],
                       'use_standardized_sub_df':False, 
                       'isDebug':False, 
                       'check_level': ['1d', '30m'], #['1d', '30m'] ['5d', '1d']
                       'sub_level_max_length':400 # 240 1200 
                       })

# mbt.prepare_initial_training_data(initial_path='./training_data/weekly_data') # change the class variable in ML_kbar_prep

# F:/A_share_chan_ml_data/201804-839-nomacd-nosubBLprocess/base_data/training_index_old.pkl
#C:/Users/MetalInvest/Desktop/ML/201804-839-1200-nomacd-nosubBLprocess/base_data/training_index_old.pkl

mbt.initial_training(initial_data_path=['./training_data/base_data/training_index.pkl'],
                     model_name='./training_model/cnn_lstm_model_index.h5',
                     epochs=8, 
                     use_ccnlstm=True)
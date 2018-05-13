########## Index Model #############


from utility.ML_kbar_prep import *
from utility.ML_model_prep import *


# from tensorflow.python.client import device_lib
# 
# print(device_lib.list_local_devices())

mld = MLDataPrep(isAnal=True)
# index_list = ['000016.XSHG','000905.XSHG','399300.XSHE', '000001.XSHG', '399001.XSHE', '399333.XSHE', '399006.XSHE']
# mld.retrieve_stocks_data(stocks=index_list,period_count=2500, filename='training_data/base_data/training_index.pkl')

filenames = ['C:/Users/MetalInvest/Desktop/ML/201805-839-1200-nomacd-subBLprocess/base_data/training_index.pkl']
x_train, x_test, y_train, y_test = mld.prepare_stock_data_cnn(filenames)
  
mdp = MLDataProcess(model_name='./training_model/cnn_lstm_model_index.h5')
mdp.define_conv_lstm_model(x_train, x_test, y_train, y_test, num_classes=3, batch_size=50 ,epochs=5, verbose=1)

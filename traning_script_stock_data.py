######### Mass Training Data #########

from ML_kbar_prep import *
from ML_model_prep import *

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n): # restart
        yield [ndx,min(ndx + n, l)]

mld = MLDataPrep(isAnal=True)
index_list = ['000016.XSHG','399300.XSHE', '399333.XSHE', '000905.XSHG', '399673.XSHE'] 
index_stocks = []
for index in index_list:
    stocks = get_index_stocks(index) #000016.XSHG 000905.XSHG 399300.XSHE
    index_stocks = index_stocks + stocks
index_stocks = list(set(index_stocks))
index_stocks.sort()
 #0
for x in batch(range(0, len(index_stocks)), 20): #50
    stocks = index_stocks[x[0]:x[1]]
#     print(x)
    print(stocks)
    mld.retrieve_stocks_data(stocks=stocks,period_count=2500, filename='training_data/training_{0}.pkl'.format(str(x[1])))

# filenames = ['training_data/cnn_training_test_index_v3_list.pkl']
# x_train, x_test, y_train, y_test = mld.prepare_stock_data_cnn(filenames)

# mdp = MLDataProcess(model_name='training_data/cnn_model_index_test.h5')
# mdp.define_conv2d_model(x_train, x_test, y_train, y_test, num_classes=3, epochs=50)

# mdp = MLDataProcess(model_name='training_data/cnn_lstm_model_index_test.h5')
# mdp.define_conv_lstm_model(x_train, x_test, y_train, y_test, num_classes=3, epochs=50)
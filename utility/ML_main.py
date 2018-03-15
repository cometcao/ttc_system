'''
Created on 28 Feb 2018

@author: MetalInvest
'''
try:
    from rqdatac import *
except:
    pass
try:
    from jqdata import *
except:
    pass


from ML_kbar_prep import *
from ML_model_prep import *
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':
    pass

#################### TRAINING ######################

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n): # 500
        yield [ndx,min(ndx + n, l)]


class ML_biaoli_train(object):
    def __init__(self, params):
        self.index_list = params.get('index_list', ['000016.XSHG','399300.XSHE', '399333.XSHE', '000905.XSHG', '399673.XSHE'])
        self.rq = params.get('rq', False)

    def prepare_mass_training_data(self, folder_path='./training_data'):
        mld = MLDataPrep(isAnal=False, rq=self.rq)
        index_stocks = []
        for index in self.index_list:
            stocks = get_index_stocks(index) #000016.XSHG 000905.XSHG 399300.XSHE
            index_stocks = index_stocks + stocks
        index_stocks = list(set(index_stocks))
        
        for x in batch(range(0, len(index_stocks)), 20): #
            stocks = index_stocks[x[0]:x[1]]
            print(stocks)
            mld.retrieve_stocks_data(stocks=stocks,period_count=2500, filename='{0}/training_{1}.pkl'.format(folder_path,str(x[1])))

    def prepare_initial_training_data(self, initial_path):
        mld = MLDataPrep(isAnal=False, rq=self.rq)
        mld.retrieve_stocks_data(stocks=self.index_list,period_count=2500, filename='{0}/training_index.pkl'.format(initial_path))

    def initial_training(self, initial_data_path, model_name, epochs=10):
        mld = MLDataPrep(isAnal=False, rq=self.rq)
        x_train, x_test, y_train, y_test = mld.prepare_stock_data_cnn(initial_data_path)
        
        mdp = MLDataProcess(model_name=model_name)
        mdp.define_conv_lstm_model(x_train, x_test, y_train, y_test, num_classes=3, epochs=epochs)        
        
        # filenames = ['training_data/cnn_training_test_index_v3_list.pkl']
        # x_train, x_test, y_train, y_test = mld.prepare_stock_data_cnn(filenames)
        
        # mdp = MLDataProcess(model_name='training_data/cnn_model_index_test.h5')
        # mdp.define_conv2d_model(x_train, x_test, y_train, y_test, num_classes=3, epochs=50)
        
        # mdp = MLDataProcess(model_name='training_data/cnn_lstm_model_index_test.h5')
        # mdp.define_conv_lstm_model(x_train, x_test, y_train, y_test, num_classes=3, epochs=50)

    def continue_training(self, model_name, folder_path='./training_data'):
        mld = MLDataPrep(isAnal=False, rq=self.rq)
        mdp = MLDataProcess(model_name=None)
        mdp.load_model(model_name)
        filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        for file in filenames:
            x_train, x_test, y_train, y_test = mld.prepare_stock_data_cnn(['{0}/{1}'.format(folder_path, file)])
            x_train = np.expand_dims(x_train, axis=2) 
            x_test = np.expand_dims(x_test, axis=2) 
        
            x_train = np.expand_dims(x_train, axis=1)
            x_test = np.expand_dims(x_test, axis=1)
            mdp.process_model(mdp.model, x_train, x_test, y_train, y_test, epochs=5)


#################### PREDICTION ######################

class ML_biaoli_check(object):
    """use CNNLSTM to predict biaoli level"""
    def __init__(self, params):
        self.threthold = params.get('threthold', 0.9)
        self.model_path = params.get('model_path', 'training_model/cnn_lstm_model_base.h5')
        self.rq = params.get('rq', True)
        self.isAnal = params.get('isAnal', False)
        self.extra_training = params.get('extra_training', False)
        self.prepare_model()

    
    def prepare_model(self):
        self.mdp = MLDataProcess(model_name=self.model_path, isAnal=self.isAnal)
        self.mdp.load_model(model_name=self.model_path)
        self.mdp.model_name = None # we don't want to save the modified model
        
    def gauge_stocks(self, stocks, isLong=True):
        if not stocks:
            return []
        if self.extra_training:
            mld = MLDataPrep(isAnal=self.isAnal, rq=self.rq)
            self.data, self.label = mld.retrieve_stocks_data(stocks, period_count=120, filename=None)
        return [stock for stock in stocks if (self.gauge_long(stock) if isLong else self.gauge_short(stock))]
        
    def gauge_long(self, stock):
        y_class, pred = self.model_predict(stock)
        conf = self.interpret(pred)
        return conf[-1] and len(y_class) >= 2 and (y_class[-1] == 1 or (y_class[-2] == -1 and y_class[-1] == 0))
        
    def gauge_short(self, stock):
        y_class, pred = self.model_predict(stock)
        conf = self.interpret(pred)
        return conf[-1] and len(y_class) >= 2 and (y_class[-1] == 1 or (y_class[-2] == 1 and y_class[-1] == 0))
        
    def model_predict(self, stock):
        mld = MLDataPrep(isAnal=self.isAnal, rq=self.rq)
        data_set = mld.prepare_stock_data_predict(stock) # 000001.XSHG
        
        if self.extra_training:
            x_train, x_test, y_train, y_test = mld.prepare_stock_data_set(self.data, self.label)
            x_train, x_test = self.mdp.define_conv_lstm_dimension(x_train, x_test)
            self.mdp.process_model(self.mdp.model, x_train, x_test, y_train, y_test, batch_size = 50,epochs = 5)
        
        unique_index = np.array([-1, 0, 1])
        return self.mdp.model_predict_cnn_lstm(data_set, unique_index)
    
    def interpret(self, pred):
        """Our confidence level must be above the threthold"""
        max_val = np.max(pred, axis=1)
        return max_val > self.threthold
######## Mass Training #########

from utility.ML_kbar_prep import *
from utility.ML_model_prep import *
from os import listdir
from os.path import isfile, join

data_dir = 'C:/Users/MetalInvest/Desktop/ML/201805-839-1200-nomacd-subBLprocess/'

mld = MLDataPrep(isAnal=False)

mdp = MLDataProcess(model_name=None, isAnal=True)
mdp.load_model('./training_model/cnn_lstm_model_index.h5')
mdp.model_name = './training_model/cnn_lstm_model_base.h5'
# mdp.load_model('C:/Users/MetalInvest/git/ttc_system/training_model/cnn_lstm_model_index.h5')

filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
filenames.sort()
for file in filenames:
    x_train, x_test, y_train, y_test = mld.prepare_stock_data_cnn(['{0}/{1}'.format(data_dir,file)])
    x_train = np.expand_dims(x_train, axis=2) 
    x_test = np.expand_dims(x_test, axis=2) 

    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)
    mdp.process_model(mdp.model, x_train, x_test, y_train, y_test, epochs=3, batch_size=5, verbose=1)
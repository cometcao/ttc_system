######## Mass Training #########

from utility.ML_kbar_prep import *
from utility.ML_model_prep import *
from os import listdir
from os.path import isfile, join
import pickle

# data_dir = 'C:/Users/MetalInvest/Desktop/ML/201805-839-nomacd-subBLprocess/'
data_dir = 'C:/Users/MetalInvest/Desktop/ML/201804-839-nomacd-nosubBLprocess/'
# data_dir = 'C:/Users/MetalInvest/Desktop/ML/201808-839-nomacd-nosubBLprocess-week/'
# data_dir = './training_data/week_data/'

record_file_path = './file_record.pkl'
try:
    file_record = load(open(record_file_path, 'rb'))
except:
    file_record = []


mld = MLDataPrep(isAnal=True,                 
                 rq=False, 
                 ts=False,
                 use_standardized_sub_df=False,
                 isDebug=False, 
                 max_length_for_pad=1200) #1200 240

mdp = MLDataProcess(model_name=None, isAnal=True)
####################
# mdp.load_model('./training_model/nosubprocessed/cnn_lstm_model_index.h5')
# mdp.model_name = './training_model/cnn_lstm_model_base.h5'

# mdp.load_model('./training_model/weekly_model/cnn_lstm_model_index_weekly.h5')
# mdp.model_name = './training_model/weekly_model/cnn_lstm_model_base_weekly.h5'

# mdp.load_model('./training_model/weekly_model/cnn_model_index_weekly.h5')
# mdp.model_name = './training_model/weekly_model/cnn_model_base_weekly.h5'
####################
mdp.load_model('./training_model/nosubprocessed/cnn_lstm_model_base.h5')
# mdp.load_model('./training_model/weekly_model/cnn_lstm_model_base_weekly.h5')
####################

filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
filenames.sort()
full_names = []
for file in filenames[10:20]:
    if file in file_record:
        continue
     
    print(file)
    full_names.append('{0}/{1}'.format(data_dir,file))
 
 
x_train, x_test, y_train, y_test = mld.prepare_stock_data_cnn(full_names)
x_train = np.expand_dims(x_train, axis=2) 
x_test = np.expand_dims(x_test, axis=2) 
 
if True:
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)
  
mdp.process_model(mdp.model, x_train, x_test, y_train, y_test, epochs=3, batch_size=50, verbose=2)


## separate file processing
#     x_train, x_test, y_train, y_test = mld.prepare_stock_data_cnn(['{0}/{1}'.format(data_dir,file)])
#     x_train = np.expand_dims(x_train, axis=2) 
#     x_test = np.expand_dims(x_test, axis=2) 
# 
#     if True:
#         x_train = np.expand_dims(x_train, axis=1)
#         x_test = np.expand_dims(x_test, axis=1)
#     
#     mdp.process_model(mdp.model, x_train, x_test, y_train, y_test, epochs=4, batch_size=100, verbose=2)
#        
#     file_record.append(file)
#     dump(file_record, open(record_file_path, 'wb'))
#######################################

# mdp.model.save_weights('./training_model/weekly_model/cnn_lstm_model_base_weight.h5')





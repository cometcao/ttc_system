# -*- encoding: utf8 -*-
######## Mass Training #########

from utility.ML_kbar_prep import *
from utility.ML_model_prep import *
from os import listdir
from os.path import isfile, join
import pickle


# data_dir = 'F:/A_share_chan_ml_data/201804-839-nomacd-nosubBLprocess/'
# data_dir = 'F:/A_share_chan_ml_data/201808-839-nomacd-nosubBLprocess-week/'
# data_dir = 'F:/A_share_chan_ml_data/201809-all-nomacd-nosubBLprocess/'
# train_data_dir = 'F:/A_share_chan_ml_data/all-process/train/'
# test_data_dir =  'F:/A_share_chan_ml_data/all-process/test/'
train_data_dir = 'F:/A_share_chan_ml_data/test-process/train/'
test_data_dir =  'F:/A_share_chan_ml_data/test-process/test/'
# train_data_dir = 'F:/A_share_chan_ml_data/check-process/train/'
# test_data_dir =  'F:/A_share_chan_ml_data/check-process/test/'
# data_dir = './training_data/week_data/'

try:
    file_record = load(open(record_file_path, 'rb'))
except:
    file_record = []


mld = MLDataPrep(isAnal=True,
                 rq=False,
                 ts=False,
                 use_standardized_sub_df=True,
                 isDebug=False, 
                 max_length_for_pad=400) #1200 240

mdp = MLDataProcess(model_name=None, isAnal=True)
####################
# mdp.load_model('./training_model/nosubprocessed/cnn_lstm_model_index.h5')
mdp.load_model('./training_model/subprocessed/cnn_lstm_model_base.h5')
# mdp.load_model('./training_model/cnn_lstm_model_base.h5')
mdp.model_name = './training_model/subprocessed/cnn_lstm_model_base2.h5'

# mdp.load_model('./training_model/weekly_model/cnn_lstm_model_index_weekly.h5')
# mdp.model_name = './training_model/weekly_model/cnn_lstm_model_base_weekly.h5'

# mdp.load_model('./training_model/weekly_model/cnn_model_index_weekly.h5')
# mdp.model_name = './training_model/weekly_model/cnn_model_base_weekly.h5'
####################
# mdp.load_model('./training_model/nosubprocessed/cnn_lstm_model_base.h5')
# mdp.load_model('./training_model/weekly_model/cnn_lstm_model_base_weekly.h5')
####################

train_filenames = [join(train_data_dir, f) for f in listdir(train_data_dir) if isfile(join(train_data_dir, f))]
train_filenames.sort()

test_filenames = [join(test_data_dir, f) for f in listdir(test_data_dir) if isfile(join(test_data_dir, f))]
test_filenames.sort()

data_gen = mld.prepare_stock_data_cnn_gen(train_filenames,
                                batch_size=100,
                                background_data_generation=False)

validation_gen = mld.prepare_stock_data_cnn_gen(test_filenames,
                                batch_size=100,
                                background_data_generation=False)
    
# mdp.define_conv_lstm_model_gen(data_gen, validation_gen, num_classes=3, epochs=8, verbose=2, steps=22500, batch_size=100, validation_steps=1000)

mdp.process_model_generator(model=mdp.model, 
                                generator=data_gen, 
                                validation_data=validation_gen, 
                                epochs=1, verbose=1, 
                                evaluate_generator=validation_gen, 
                                steps=410000, validation_steps=100000)

# mdp.model.save_weights('./training_model/weekly_model/cnn_lstm_model_base_weight.h5')


# print(daily_avg)
# print(daily_max)
# print(weekly_avg)
# print(weekly_max)
# 11.3079761885 (1d)
# 89.0
# 11.0538684439 (5d)
# 49.0



# -*- encoding: utf8 -*-
from utility.ML_main import *
from utility.ML_kbar_prep import *
from utility.ML_model_prep import *

stock = '510050.XSHG'

mbt = ML_biaoli_train({'ts':False,
                       'rq':False, 
                       'isAnal':True,
                       'use_standardized_sub_df':False})

mbt.specified_stock_training(stocks=[stock], model_name='./training_model/nosubprocessed/cnn_lstm_model_base.h5', period_count=750, 
                   training_data_path='./training_data/',
                   batch_size=5, epochs=30, detailed_bg=True)


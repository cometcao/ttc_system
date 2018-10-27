# -*- encoding: utf8 -*-
######### Mass Training Data #########
########  Turning Raw Data into training data ########

from utility.ML_kbar_prep import *
from os import listdir
from os.path import isfile, join
import pickle

pd.options.mode.chained_assignment = None

mld = MLDataPrep(isAnal=True, 
                 rq=False, 
                 ts=False,
                 use_standardized_sub_df=True,
                 isDebug=False, 
                 monitor_level=['1d', '30m'],
                 max_length_for_pad=400)

raw_file_dir = "F:/A_share_chan_ml_data/201810-840-raw/"
process_file_path = "F:/A_share_chan_ml_data/201810-840-nomacd-subprocesssplit/"

rawfilenames = [join(raw_file_dir, f) for f in listdir(raw_file_dir) if isfile(join(raw_file_dir, f))]
i_index = 1
for raw_file in rawfilenames:
    mld.retrieve_stocks_data_from_raw(raw_file_path=raw_file, filename='{0}/training_{1}.pkl'.format(process_file_path, i_index))
    i_index += 1
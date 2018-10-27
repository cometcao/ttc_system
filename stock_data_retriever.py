######## Data Grabber #########
######## get stock raw data ########

from utility.ML_kbar_prep import *
from os import listdir
from os.path import isfile, join
from pickle import dump
from pickle import load
from utility.common_include import batch
from utility.securityDataManager import JqDataRetriever


data_dir = 'F:/A_share_chan_ml_data/201810-840-nomacd-subprocesssplit/'
save_data_dir = 'F:/A_share_chan_ml_data/201810-840-raw/'

mlk = MLKbarPrep(isAnal=True, 
                 count=2500, 
                 isNormalize=True, 
                 sub_max_count=400, 
                 isDebug=False, 
                 sub_level_min_count=0, 
                 use_standardized_sub_df=True,
                 monitor_level=['1d', '30m'])

index_list = ['000016.XSHG','000905.XSHG','399300.XSHE', '000001.XSHG', '399001.XSHE', '399333.XSHE', '399006.XSHE']

index_stocks = []
for index in index_list:
    stocks = JqDataRetriever.get_index_stocks(index) 
    index_stocks = index_stocks + stocks
index_stocks = list(set(index_stocks))
index_stocks.sort()

for i in batch(range(0, len(index_stocks)), 200):
    mlk.grab_stocks_raw_data(index_stocks[i[0]:i[1]], file_dir=save_data_dir)
# mlk.load_stock_raw_data("{0}/{1}".format(save_data_dir, "000001.XSHE.pkl"))
# 
# print(mlk.stock_df_dict)


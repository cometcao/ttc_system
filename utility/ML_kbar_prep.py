# -*- encoding: utf8 -*-
try:
    from rqdatac import *
except:
    pass
try:
    from jqdata import *
except:
    pass
from utility.common_include import *
from utility.kBarProcessor import *
from utility.biaoLiStatus import TopBotType
from keras.utils.np_utils import to_categorical
from pickle import dump
from pickle import load
import pandas as pd
import numpy as np
import talib
import datetime
from sklearn.model_selection import train_test_split
from utility.securityDataManager import *
from utility.utility_ts import *

# pd.options.mode.chained_assignment = None 

fixed_length = 1200

# # save a dataset to file
# def save_dataset(dataset, filename):
#     dump(dataset, open(filename, 'wb'))
# #         put_file(filename, dataset, append=False)
#     print('Saved: %s' % filename)
#     
# # load a clean dataset
# def load_dataset(filename):
#     print("Loading file: {0}".format(filename))
#     return load(open(filename, 'rb'))
# #         return get_file(filename)


class MLKbarPrep(object):
    '''
    Turn multiple level of kbar data into Chan Biaoli status,
    return a dataframe with combined biaoli status
    data types:
    biaoli status, high/low prices, volume/turnover ratio/money, MACD, sequence index
    '''

    def __init__(self, count=100, 
                 isAnal=False, 
                 isNormalize=True, 
                 manual_select=False, 
                 useMinMax=True, 
                 sub_max_count=fixed_length, 
                 isDebug=False, 
                 include_now=False, 
                 sub_level_min_count = 0, 
                 use_standardized_sub_df=False,
                 monitor_level = ['1d', '30m']):
        self.isDebug = isDebug
        self.isAnal = isAnal
        self.count = count
        self.isNormalize = isNormalize
        self.useMinMax = useMinMax
        self.manual_select = manual_select
        self.stock_df_dict = {}
        self.sub_level_min_count = sub_level_min_count
        self.sub_max_count = sub_max_count
        self.data_set = []
        self.label_set = []
        self.include_now = include_now
        self.use_standardized_sub_df = use_standardized_sub_df
        self.num_of_debug_display = 4
        self.monitor_level = monitor_level

    def workout_count_num(self, level):
        return self.count if self.monitor_level[0] == level \
                        else self.count * 8 if self.monitor_level[0] == '1d' and level == '30m' \
                        else self.count * 8 if self.monitor_level[0] == '5d' and level == '150m' \
                        else self.count * 24 if self.monitor_level[0] == '5d' and level == '30m' \
                        else self.count * 8

    def get_high_df(self):
        return self.stock_df_dict[self.monitor_level[0]]
    
    def get_low_df(self):
        return self.stock_df_dict[self.monitor_level[1]]

    def grab_stock_raw_data(self, stock, end_date, fields=['open','close','high','low', 'money'], file_dir="."):
        temp_stock_df_dict = {}
        for level in self.monitor_level:
            local_count = self.workout_count_num(level)
            stock_df = None
            if not self.isAnal:
                stock_df = attribute_history(stock, local_count, level, fields = fields, skip_paused=True, df=True)  
            else:
                latest_trading_day = str(end_date if end_date is not None else datetime.datetime.today().date())
                latest_trading_day = latest_trading_day+" 15:00:00" if level == '30m' else latest_trading_day # hack for get_price to get latest 30m data change back to 14:30:00
                stock_df = SecurityDataManager.get_research_data_jq(stock, count=local_count, end_date=latest_trading_day, period=level, fields = fields, skip_suspended=True)          
            if stock_df.empty:
                continue
            temp_stock_df_dict[level] = stock_df
        return temp_stock_df_dict
        
    def grab_stocks_raw_data(self, stocks, end_date=None, fields=['open','close','high','low', 'money'], file_dir="."):
        # grab the raw data and save on files
        all_stock_df = []
        for stock in stocks:
            all_stock_df.append(self.grab_stock_raw_data(stock, end_date, fields, file_dir))
        save_dataset(all_stock_df, "{0}/last_stock_{1}.pkl".format(file_dir, stocks[-1]))

    def load_stock_raw_data(self, stock_df):
        self.stock_df_dict = stock_df
        for level in self.monitor_level:
            self.stock_df_dict[level] = self.prepare_df_data(self.stock_df_dict[level], level)
        
    
    def retrieve_stock_data(self, stock, end_date=None):
        for level in self.monitor_level:
            local_count = self.workout_count_num(level)
            stock_df = None
            if not self.isAnal:
                stock_df = attribute_history(stock, local_count, level, fields = ['open','close','high','low', 'money'], skip_paused=True, df=True)  
            else:
                latest_trading_day = str(end_date if end_date is not None else datetime.datetime.today().date())
                latest_trading_day = latest_trading_day+" 14:30:00" if level == '30m' else latest_trading_day # hack for get_price to get latest 30m data
                stock_df = SecurityDataManager.get_research_data_jq(stock, count=local_count, end_date=latest_trading_day, period=level, fields = ['high','low', 'money'], skip_suspended=True)          
            if stock_df.empty:
                continue
#             if self.isDebug:
#                 print("{0}, {1}, {2}, {3}".format(stock, local_count, end_date, level))
#                 print(stock_df.tail(self.num_of_debug_display))
            stock_df = self.prepare_df_data(stock_df, level)
            self.stock_df_dict[level] = stock_df
    
    def retrieve_stock_data_rq(self, stock, end_date=None):
        for level in self.monitor_level:
            stock_df = None
            if not self.isAnal:
                local_count = self.workout_count_num(level)
                stock_df = SecurityDataManager.get_data_rq(stock, count=local_count, period=level, fields=['open','close','high','low', 'total_turnover'], skip_suspended=True, df=True, include_now=self.include_now)
            else:
                today = end_date if end_date is not None else datetime.datetime.today()
                previous_trading_day=get_trading_dates(start_date='2006-01-01', end_date=today)[-self.count]
                stock_df = SecurityDataManager.get_research_data_rq(stock, start_date=previous_trading_day, end_date=today, period=level, fields = ['open','close','high','low', 'total_turnover'], skip_suspended=True)
            if stock_df.empty:
                continue
            stock_df = self.prepare_df_data(stock_df, level)
            self.stock_df_dict[level] = stock_df    
        
    def retrieve_stock_data_ts(self, stock, end_date=None):
        today = end_date if end_date is not None else datetime.datetime.today()
        previous_trading_day=get_trading_date_ts(count=self.count, end=today)[-self.count]
        for level in self.monitor_level:
            ts_level = 'D' if level == '1d' else '30' if level == '30m' else 'D' # 'D' as default
            stock_df = SecurityDataManager.get_data_ts(stock, start_date=previous_trading_day, end_date=today, period=ts_level)
            if stock_df is None or stock_df.empty:
                continue
            stock_df = self.prepare_df_data(stock_df, level)
            self.stock_df_dict[level] = stock_df
    
    def prepare_df_data(self, stock_df, level):
        # MACD # don't use it now
#         stock_df.loc[:,'macd_raw'], _, stock_df.loc[:,'macd']  = talib.MACD(stock_df['close'].values)
        stock_df = stock_df.dropna() # make sure we don't get any nan data
        stock_df = self.prepare_biaoli(stock_df, level)
        return stock_df
        
    
    def prepare_biaoli(self, stock_df, level):
        if level == self.monitor_level[0]:
            kb = KBarProcessor(stock_df)
            # for higher level, we only need the pivot dates, getMarketBL contains more than we need, no need for join
            stock_df = kb.getMarkedBL()

        elif level == self.monitor_level[1]:
            if self.use_standardized_sub_df:
                kb = KBarProcessor(stock_df)
                if self.sub_level_min_count != 0:
                    stock_df = kb.getMarkedBL()[['open','close','high','low', 'money', 'new_index', 'tb']]
                else:
                    # stock_df = kb.getStandardized()[['open','close','high','low', 'money']]
                    # logic change here use sub level pivot time for segmentation of background training data
                    stock_df = kb.getIntegraded()
            else:
                pass
        return stock_df
    
    def prepare_training_data(self):
        if len(self.stock_df_dict) == 0:
            return [], []
        higher_df = self.stock_df_dict[self.monitor_level[0]]
        lower_df = self.stock_df_dict[self.monitor_level[1]]
        high_df_tb = higher_df.dropna(subset=['new_index'])
        high_dates = high_df_tb.index
        for i in range(0, len(high_dates)-1):
            first_date = str(high_dates[i].date())
            second_date = str(high_dates[i+1].date())
            trunk_lower_df = lower_df.loc[first_date:second_date,:]
            self.create_ml_data_set(trunk_lower_df, high_df_tb.ix[i+1, 'tb'].value, for_predict=False)
        return self.data_set, self.label_set
    
    def prepare_predict_data(self):    
        higher_df = self.stock_df_dict[self.monitor_level[0]]
        lower_df = self.stock_df_dict[self.monitor_level[1]]
        high_df_tb = higher_df.dropna(subset=['new_index'])
        if self.isDebug:
            if high_df_tb.shape[0] > self.num_of_debug_display:
                print(high_df_tb.tail(self.num_of_debug_display)[['tb', 'new_index']])
            else:
                print(high_df_tb[['tb', 'new_index']])
        high_dates = high_df_tb.index
        
        for i in range(-self.num_of_debug_display-1, 0, 1): #-5
            try:
                previous_date = str(high_dates[i].date())
            except IndexError:
                continue
            trunk_df = None
            if i+1 < 0:
                next_date = str(high_dates[i+1].date())
                trunk_df = lower_df.loc[previous_date:next_date, :]
            else:
                trunk_df = lower_df.loc[previous_date:, :]
#             if self.isDebug:
#                 print(trunk_df.tail(self.num_of_debug_display))
            self.create_ml_data_set(trunk_df, None, for_predict=True)
        return self.data_set
               
    def prepare_predict_data_extra(self):
        higher_df = self.stock_df_dict[self.monitor_level[0]]
        lower_df = self.stock_df_dict[self.monitor_level[1]]
        high_df_tb = higher_df.dropna(subset=['new_index'])
        high_dates = high_df_tb.index
        # additional check trunk
        for i in range(-self.num_of_debug_display-1, -1, 2):#-5
            try:
                previous_date = str(high_dates[i].date())
            except IndexError:
                continue
            trunk_df = lower_df.loc[previous_date:,:]
#             if self.isDebug:
#                 print(trunk_df.head(self.num_of_debug_display))
#                 print(trunk_df.tail(self.num_of_debug_display))
            self.create_ml_data_set(trunk_df, None, for_predict=True)
        return self.data_set
        
    def create_ml_data_set(self, trunk_df, label, for_predict=False): 
        # at least 3 parts in the sub level
        if not for_predict and self.sub_level_min_count != 0: # we won't process sub level df
            sub_level_count = len(trunk_df['tb']) - trunk_df['tb'].isnull().sum()
            if sub_level_count < self.sub_level_min_count:
                return
        
#         copy_trunk_df = trunk_df.copy(deep=True)
        # sub level trunks
        tb_trunk_df = trunk_df.dropna(subset=['tb'])
        
        if len(tb_trunk_df.index) >= 2: # precise sub level chunk at least 2 subs
            trunk_df = trunk_df.loc[tb_trunk_df.index[0]:tb_trunk_df.index[-1],:]
        
        if self.sub_max_count > 0 and trunk_df.shape[0] > self.sub_max_count: # truncate
            print("data truncated due to max length sequence exceeded")
            trunk_df = trunk_df.iloc[-self.sub_max_count:,:]
        
        if self.manual_select:
            trunk_df = self.manual_select(trunk_df)
        else: # manual_wash
            trunk_df = self.manual_wash(trunk_df)  
        if self.isNormalize:
            trunk_df = self.normalize(trunk_df)
        
#         if trunk_df.isnull().values.any():
#             print("NaN value found, ignore this data")
#             print(trunk_df)
#             print(copy_trunk_df)
#             print(tb_trunk_df)
#             return
    
        if for_predict: # differentiate training and predicting
            self.data_set.append(trunk_df.values)
        else:
            self.data_set.append(trunk_df.values)
            self.label_set.append(label)
            
            for time_index in tb_trunk_df.index[1:-2]: #  counting from cutting start
                self.data_set.append(trunk_df.loc[:time_index, :].values)
                self.label_set.append(TopBotType.noTopBot.value)
        
        
    def manual_select(self, df):
        df = df.dropna() # only concern BI
        df['new_index'] = df['new_index'].shift(-1) - df['new_index'] 
        df['tb'] = df.apply(lambda row: row['tb'].value, axis = 1)
        df['price'] = df.apply(lambda row: row['high'] if row['tb'] == 1 else row['low'])
        df.drop(['open', 'high', 'low'], 1)
        return df
        
    def manual_wash(self, df):
#         if self.sub_level_min_count != 0:
        df = df.drop(['new_index','tb'], 1, errors='ignore')
#         df = df.dropna() 
        return df
        
    def normalize(self, df):
        for column in df: 
            if column == 'new_index' or column == 'tb':
                continue
            if self.useMinMax:
                # min-max -1 1 / 0 1
                col_min = df[column].min()
                col_max = df[column].max()
                col_mean = df[column].mean()
                df[column]=(df[column]-col_mean)/(col_max-col_min)
            else:
                # use zscore mean std
                col_mean = df[column].mean()
                col_std = df[column].std()
                df[column] = (df[column] - col_mean) / col_std
        return df


class MLDataPrep(object):
    def __init__(self, isAnal=False, max_length_for_pad=fixed_length, rq=False, ts=True, useMinMax=False, isDebug=False,detailed_bg=False, use_standardized_sub_df=True, monitor_level=['1d','30m']):
        self.isDebug = isDebug
        self.isAnal = isAnal
        self.detailed_bg = detailed_bg
        self.max_sequence_length = max_length_for_pad
        self.isRQ = rq
        self.isTS = ts
        self.unique_index = []
        self.use_standardized_sub_df = use_standardized_sub_df
        self.check_level = monitor_level
        self.useMinMax = useMinMax
    
    def retrieve_stocks_data_from_raw(self, raw_file_path=None, filename=None):
        data_list = []
        label_list = []
        mlk = MLKbarPrep(isAnal=self.isAnal, 
                         isNormalize=True, 
                         sub_max_count=self.max_sequence_length, 
                         useMinMax=self.useMinMax,
                         isDebug=self.isDebug, 
                         sub_level_min_count=0, 
                         use_standardized_sub_df=self.use_standardized_sub_df, 
                         monitor_level=self.check_level)

        df_array = load_dataset(raw_file_path)
        for stock_df in df_array:
            mlk.load_stock_raw_data(stock_df)
            dl, ll = mlk.prepare_training_data()
            
            data_list = data_list + dl
            label_list = label_list + ll
        if filename:
            save_dataset((data_list, label_list), filename)
        return (data_list, label_list)            
    
    def retrieve_stocks_data(self, stocks, period_count=60, filename=None, today_date=None):
        data_list = []
        label_list = []
        for stock in stocks:
            if self.isAnal:
                print ("working on stock: {0}".format(stock))
            mlk = MLKbarPrep(isAnal=self.isAnal, 
                             count=period_count, 
                             isNormalize=True, 
                             sub_max_count=self.max_sequence_length, 
                             useMinMax=self.useMinMax,
                             isDebug=self.isDebug, 
                             sub_level_min_count=0, 
                             use_standardized_sub_df=self.use_standardized_sub_df, 
                             monitor_level=self.check_level)
            if self.isTS:
                mlk.retrieve_stock_data_ts(stock, today_date)
            elif self.isRQ:
                mlk.retrieve_stock_data_rq(stock, today_date)
            else:
                mlk.retrieve_stock_data(stock, today_date)
            dl, ll = mlk.prepare_training_data()
            data_list = data_list + dl
            label_list = label_list + ll   
        if filename:
            save_dataset((data_list, label_list), filename)
        return (data_list, label_list)
    
    def prepare_stock_data_predict(self, stock, period_count=100, today_date=None):
        mlk = MLKbarPrep(isAnal=self.isAnal, 
                         count=period_count, 
                         isNormalize=True, 
                         sub_max_count=self.max_sequence_length, 
                         useMinMax=self.useMinMax,
                         isDebug=self.isDebug, 
                         sub_level_min_count=0, 
                         use_standardized_sub_df=self.use_standardized_sub_df,
                         monitor_level=self.check_level)
        if self.isTS:
            mlk.retrieve_stock_data_ts(stock, today_date)
        elif self.isRQ:
            mlk.retrieve_stock_data_rq(stock, today_date)
        else:
            mlk.retrieve_stock_data(stock, today_date)
        predict_dataset = mlk.prepare_predict_data()
        origin_pred_size = len(predict_dataset)
        if origin_pred_size == 0:
            return None, 0
        predict_dataset = mlk.prepare_predict_data_extra()
        
        predict_dataset = pad_each_training_array(predict_dataset, self.max_sequence_length)
        if self.isDebug:
#             print("original size:{0}".format(origin_pred_size))
            pass
        return predict_dataset, origin_pred_size, mlk.get_high_df().ix[-1, 'tb'].value
        
    def encode_category(self, label_set):
        uniques, ids = np.unique(label_set, return_inverse=True)
        y_code = to_categorical(ids, len(uniques))
        self.unique_index = uniques
        return y_code
    
    def prepare_stock_data_cnn(self, filenames, padData=True, test_portion=0.1, random_seed=42, background_data_generation=False):
        data_list = []
        label_list = []
        for file in filenames:
            A, B = load_dataset(file)
            
            A_check = True
            i = 0
            for item in A:     
                if (self.useMinMax and not ((item>=-1).all() and (item<=1).all())) or np.isnan(item).any(): # min max value range
                    print(item)
                    print(A[i])
                    print(B[i])
                    print(i)
                    A_check=False
                    break
                i += 1
            if not A_check:
                print("Data invalid in file {0}".format(file))
                continue

            data_list = data_list + A
            label_list = label_list + B

        return self.prepare_stock_data_set(data_list, label_list, padData, test_portion, random_seed, background_data_generation)
        
    def prepare_stock_data_set(self, data_list, label_list, padData=True, test_portion=0.1, random_seed=42, background_data_generation=False):
        if not data_list or not label_list:
            print("Invalid file content")
            return

#         if self.isDebug:
#             print (data_list)
#             print (label_list)

        if background_data_generation:
            data_list, label_list = self.prepare_background_data(data_list, label_list)

        if padData:
            data_list = self.pad_each_training_array(data_list, self.max_sequence_length)
        
        label_list = self.encode_category(label_list)  
        
        x_train, x_test, y_train, y_test = train_test_split(data_list, label_list, test_size=test_portion, random_state=random_seed)
        
        if self.isDebug:
#             print (x_train.shape)
#             print (x_train)
#             print (y_train)
            pass
        
        return x_train, x_test, y_train, y_test
    
    def prepare_background_data(self, data_set, label_set):
        # split existing samples to create sample for 0 label
        split_ratio = [0.191, 0.382, 0.5, 0.618, 0.809]
        new_background_data = []
        new_label_data = []
#         print(len(data_set))
        for sample in data_set:
            length = sample.shape[0]
            if self.detailed_bg:
                for i in range(2, length-1, 2): # step by 2
                    new_data = sample[:i, :] 
                    new_background_data.append(new_data)
                    new_label_data.append(TopBotType.noTopBot.value)
            else:
                for split_index in split_ratio:
                    si = int(split_index * length)
                    new_data = sample[:si,:]
                    new_background_data.append(new_data)
                    new_label_data.append(TopBotType.noTopBot.value)
        
        data_set = data_set + new_background_data
        label_set = label_set + new_label_data
        return data_set, label_set

    def define_conv_lstm_dimension(self, x_train):
        x_train = np.expand_dims(x_train, axis=2)         #3
        x_train = np.expand_dims(x_train, axis=2)
        return x_train
    

    def generate_from_data(self, data, label, batch_size):
        for i in batch(range(0, len(data)), batch_size):
            yield data[i[0]:i[1]], label[i[0]:i[1]]    
    
    def keepDimension(self, targetArray, indexList):
        tempArray = np.array(targetArray)
        tempArray = tempArray[:,:,indexList]
        return tempArray.tolist()
    
    def generate_from_file(self, filenames, padData=True, background_data_generation=False, batch_size=50):
        while True:
            for file in filenames:
                A, B = load_dataset(file)
                
                A_check = True
                for item in A:     
                    if (self.useMinMax and not ((item>=-1).all() and (item<=1).all())) or np.isnan(item).any(): # min max value range or zscore
                        print(item)
                        A_check=False
                        break
                if not A_check:
                    print("Data invalid in file {0}".format(file))
                    continue
    
#                 print("loaded data set: {0}".format(file))
    
                if not A or not B:
                    print("Invalid file content")
                    return
    
                if background_data_generation:
                    A, B = self.prepare_background_data(A, B)
    
#                 if padData:
#                     A = pad_each_training_array(A, self.max_sequence_length)
                
                B = self.encode_category(B)
#                 A = self.define_conv_lstm_dimension(A)
                
                for i in batch(range(0, len(A)), batch_size):
                    subA = A[i[0]:i[1]]
                    
                    if padData:
                        subA = pad_each_training_array(subA, self.max_sequence_length)
                    else:
                        subA = np.array(subA)
                    
                    subA = self.define_conv_lstm_dimension(subA)
                    
                    yield subA, B[i[0]:i[1]] 
    
    def prepare_stock_data_cnn_gen(self, filenames, padData=True, background_data_generation=False, batch_size=50):
        return self.generate_from_file(filenames, padData=padData, background_data_generation=background_data_generation, batch_size=batch_size)
    
#                           open      close       high        low        money  \
# 2017-11-14 10:00:00  3446.5500  3436.1400  3450.3400  3436.1400  60749246464   
# 2017-11-14 10:30:00  3436.7000  3433.1700  3438.7300  3431.2600  39968927744   
# 2017-11-14 11:00:00  3433.3600  3437.7500  3439.4100  3429.8200  28573523968   

# 
#                       macd_raw      macd  new_index              tb  
# 2017-11-14 10:00:00   9.480639 -0.786244        NaN             NaN  
# 2017-11-14 10:30:00   8.310828 -1.564845        NaN             NaN  
# 2017-11-14 11:00:00   7.664954 -1.768575        NaN             NaN  
# 2017-11-14 11:30:00   6.671123 -2.209925        NaN             NaN  
# 2017-11-14 13:30:00   6.626142 -1.803925        NaN             NaN  
# 2017-11-14 14:00:00   6.067070 -1.890397        NaN             NaN  
# 2017-11-14 14:30:00   4.368913 -2.870843        NaN             NaN  
# 2017-11-14 15:00:00   3.564614 -2.940114        NaN             NaN  
# 2017-11-15 10:00:00   1.701251 -3.842782        NaN             NaN  
# 2017-11-15 10:30:00  -0.326071 -4.696083        NaN             NaN  
# 2017-11-15 11:00:00  -1.975328 -5.076272        NaN             NaN  
# 2017-11-15 11:30:00  -3.382178 -5.186497        NaN             NaN  
# 2017-11-15 13:30:00  -4.234472 -4.831033        NaN             NaN  
# 2017-11-15 14:00:00  -4.859551 -4.364890        NaN             NaN  
# 2017-11-15 14:30:00  -5.841940 -4.277823        NaN             NaN  
# 2017-11-15 15:00:00  -6.416611 -3.881995        NaN             NaN  
# 2017-11-16 10:00:00  -6.918969 -3.507483         51  TopBotType.bot  
# 2017-11-16 10:30:00  -7.690800 -3.423451        NaN             NaN  
# 2017-11-16 11:00:00  -7.859263 -2.873531        NaN             NaN  
# 2017-11-16 11:30:00  -7.935189 -2.359566        NaN             NaN  
# 2017-11-16 13:30:00  -8.347779 -2.217725        NaN             NaN  
# 2017-11-16 14:00:00  -7.629007 -1.199162        NaN             NaN  
# 2017-11-16 14:30:00  -7.446391 -0.813237         57  TopBotType.top  
# 2017-11-16 15:00:00  -7.247972 -0.491854        NaN             NaN  
# 2017-11-17 10:00:00  -7.885018 -0.903120        NaN             NaN  




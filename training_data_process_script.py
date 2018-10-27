# -*- encoding: utf8 -*-
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from utility.common_include import *
import sys

# data_dir = 'F:/A_share_chan_ml_data/201809-all-nomacd-nosubBLprocess/'
# save_data_dir = 'F:/A_share_chan_ml_data/all-process/'

data_dir = 'F:/A_share_chan_ml_data/201810-840-nomacd-subprocesssplit/'
save_data_dir = 'F:/A_share_chan_ml_data/test-process/'

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n): # restart
        yield [ndx,min(ndx + n, l)]

filenames = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
filenames.sort()

all_data = []
all_label = []
# gather all data
for file in filenames:
    A, B = load_dataset(file)
    all_data = all_data + A
    all_label = all_label + B

x_train, x_test, y_train, y_test = train_test_split(all_data, all_label, test_size=0.2, random_state=28)

# save all data
for i in batch(range(0, len(x_train)), 100000):
    save_dataset((x_train[i[0]:i[1]], y_train[i[0]:i[1]]), "{0}/train_{1}".format(save_data_dir, i[1])) 
 
for i in batch(range(0, len(x_test)), 50000):
    save_dataset((x_test[i[0]:i[1]], y_test[i[0]:i[1]]), "{0}/test_{1}".format(save_data_dir, i[1]))


# save_dataset_np((x_train, y_train), "{0}/train_0.pkl".format(save_data_dir))
# save_dataset_np((x_test, y_test), "{0}/test_0.pkl".format(save_data_dir))
    
# split_size = int(len(all_data) * 0.999)
# x_train = all_data[:split_size]
# x_test = all_data[split_size:]
# y_train = all_label[:split_size]
# y_test = all_label[split_size:]
# 
# print(len(all_data))
# print(len(all_label))
# print(len(x_train))
# print(len(y_train))
# print(len(x_test))
# print(len(y_test))
# print("-------------------")
# print(sys.getsizeof(all_data))
# print(sys.getsizeof(all_label))
# print(sys.getsizeof(x_train))
# print(sys.getsizeof(y_train))
# print(sys.getsizeof(x_test))
# print(sys.getsizeof(y_test))
# print("+++++++++++++++++++")

#     save_dataset((x_train, y_train), "{0}/train_{1}.pkl".format(save_data_dir, file_index))
#     save_dataset((x_test, y_test), "{0}/test_{1}.pkl".format(save_data_dir, file_index))

######## Mass Training #########

from utility.ML_kbar_prep import *
from utility.ML_model_prep import *
from os import listdir
from os.path import isfile, join
import pickle



train_data_dir = 'F:/A_share_chan_ml_data/5d-30m-process-test/'
test_data_dir =  'F:/A_share_chan_ml_data/5d-30m-process-test/'

# data_dir = './training_data/week_data/'

####################
mdp = MLDataProcess(model_name=None, isAnal=True, isDebug=True)
mdp.model_name = './training_model/subprocessed/cnn_lstm_model_base_5d30m_test.h5'
####################

def check_done(lines, cs):    
    for l in lines:
        l_p = l.split(':')
        if cs == l_p[0]:
            return True
    return False

############################# Tuning ###############################

############################# Data Processing ######################
normalization_method = [None, [0,1], [-1,1]]


############################# Model Tuning #########################
model_type = ['rnncnn','convlstm',] #'cnn',
network_topology = ['normal', 'deep']
batch_size = [10,   100] #20, 60, 40, 80, 50,
epochs_size = [55] #89,   133
hidden_neurons = [13,  144] #34, 89, 233,  377
weight_init =  ['he_uniform', 'glorot_uniform', 'he_normal', 'glorot_normal', 'lecun_uniform', 'lecun_normal'] #, 'normal', 'uniform','zero'
learning_rate =  [0.001, 0.01, 0.25] # 0.2,  0.1, 0.3
momentum = [0.0, 0.1, 0.5]#0.2,0.4, ,0.6, 0.8, 0.9, 0.3, 0.7
activation_func = ['relu',  'tanh', 'softsign'] # 'softplus', 'sigmoid', ## , 'softmax',  'hard_sigmoid', 'linear'
optimization_method = ['SGD'] #####, 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
drop_out_rate =  [0.3] #0.0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.6,  0.9
# loss_func = []

filename = "log/hyperparameter_test.txt"
lines = [line.rstrip('\n') for line in open(filename)]

for mt in model_type:
    for nm in normalization_method:
        if nm is None:
            tmp_train_data_dir = train_data_dir + "/train-z"
            tmp_test_data_dir = test_data_dir + "/test-z"
        elif nm == [0, 1]:
            tmp_train_data_dir = train_data_dir + "/train-01"
            tmp_test_data_dir = test_data_dir + "/test-01"  
        elif nm == [-1, 1]:
            tmp_train_data_dir = train_data_dir + "/train-11"
            tmp_test_data_dir = test_data_dir + "/test-11"
        train_filenames = [join(tmp_train_data_dir, f) for f in listdir(tmp_train_data_dir) if isfile(join(tmp_train_data_dir, f))]
        train_filenames.sort()
        
        test_filenames = [join(tmp_test_data_dir, f) for f in listdir(tmp_test_data_dir) if isfile(join(tmp_test_data_dir, f))]
        test_filenames.sort()
        
        mld = MLDataPrep(isAnal=True,
                         rq=False,
                         ts=False,
                         use_standardized_sub_df=True,
                         isDebug=False, 
                         norm_range=nm,
                         max_length_for_pad=268)
        for nt in network_topology:
            for bs in batch_size:
                data_gen = mld.prepare_stock_data_gen(train_filenames,
                                batch_size=bs, #100 25
                                background_data_generation=False,
                                padData=True,model_type=mt) # convlstm, rnncnn, cnn
                                                
                validation_gen = mld.prepare_stock_data_gen(test_filenames,
                                                batch_size=bs, #100 25
                                                background_data_generation=False,
                                                padData=True,model_type=mt)
                for es in epochs_size:
                    for hn in hidden_neurons:          
                        for wi in weight_init:
                            for om in optimization_method:
                                for dor in drop_out_rate: 
                                    for af in activation_func:
                                        for lr in learning_rate:
                                            for m in momentum:                                              
                                                best_val_loss = best_val_acc = 0.0
                                                hp= HyperParam(nt, hn, wi, lr, m, af, dor, om)
                                                
                                                m_back = 0.0 if om != "SGD" else m # only SGD use momentum other optimizer we override to 0.0
                                                
                                                check_str = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}".format(mt, nt, bs, es, hn, wi, lr, m_back, af, om, dor, nm)
                                                if check_done(lines, check_str):
                                                    print("{0} already done".format(check_str))
                                                    continue
                                                else:
                                                    print("{0} not done".format(check_str))
                                                model_func = mdp.define_conv_lstm_model_gen if mt =='convlstm' else \
                                                            mdp.define_rnn_cnn_model_gen if mt == 'rnncnn' else \
                                                            mdp.define_cnn_model_gen if mt == 'cnn' else None
                                                best_val_loss, best_val_acc, best_loss, best_acc = model_func(data_gen, 
                                                                               validation_gen, 
                                                                               num_classes=3, 
                                                                               epochs=es, 
                                                                               verbose=2, 
                                                                               steps=int(4912/bs), 
                                                                               batch_size=bs, 
                                                                               validation_steps=int(1228/bs), 
                                                                               patience=int(es/6), hyper_param=hp) 
                                                                                                  
                                                result_str = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}:,{12},{13},{14},{15}\n".format(mt, nt, bs, es, hn, wi, lr, m_back, af, om, dor, nm, best_val_loss, best_val_acc, best_loss, best_acc)
                                                with open(filename, "a+") as result_file:
                                                    result_file.write(result_str)
                                                
        

######## Mass Training #########

from utility.ML_kbar_prep import *
from utility.ML_model_prep import *
from os import listdir
from os.path import isfile, join
import pickle

data_dir = 'C:/Users/MetalInvest/Desktop/ML/201805-839-1200-nomacd-subBLprocess/'

record_file_path = './file_record.pkl'
try:
    file_record = load(open(record_file_path, 'rb'))
except:
    file_record = ['training_10.pkl','training_15.pkl','training_20.pkl',
                   'training_25.pkl','training_30.pkl','training_35.pkl',
                   'training_40.pkl','training_45.pkl','training_50.pkl',
                   'training_55.pkl','training_60.pkl',

                   'training_100.pkl','training_105.pkl','training_110.pkl',
                   'training_115.pkl','training_120.pkl','training_125.pkl',
                   'training_130.pkl','training_135.pkl','training_140.pkl',
                   'training_145.pkl','training_150.pkl','training_155.pkl',
                   'training_160.pkl','training_165.pkl','training_170.pkl',
                   'training_175.pkl','training_180.pkl','training_185.pkl',
                   'training_190.pkl','training_195.pkl','training_200.pkl',
                   'training_205.pkl','training_210.pkl','training_215.pkl',
                   'training_220.pkl','training_225.pkl','training_230.pkl',
                   'training_235.pkl','training_240.pkl','training_245.pkl',
                   'training_250.pkl','training_255.pkl','training_260.pkl',
                   'training_265.pkl','training_270.pkl','training_275.pkl',
                   'training_280.pkl','training_285.pkl','training_290.pkl',
                   'training_295.pkl','training_300.pkl','training_305.pkl',
                   'training_310.pkl','training_315.pkl','training_320.pkl',
                   'training_325.pkl','training_330.pkl','training_335.pkl',
                   'training_340.pkl','training_345.pkl','training_350.pkl',
                   'training_355.pkl','training_360.pkl','training_365.pkl',
                   'training_370.pkl','training_375.pkl','training_380.pkl',
                   'training_385.pkl','training_390.pkl','training_395.pkl',
                   'training_400.pkl','training_405.pkl','training_410.pkl',
                   'training_415.pkl','training_420.pkl','training_425.pkl',
                   'training_430.pkl','training_435.pkl','training_440.pkl',
                   'training_445.pkl','training_450.pkl','training_455.pkl',
                   'training_460.pkl','training_465.pkl','training_470.pkl',
                   'training_475.pkl','training_480.pkl','training_485.pkl',
                   'training_490.pkl','training_495.pkl','training_500.pkl',
                   'training_505.pkl','training_510.pkl','training_515.pkl',
                   'training_520.pkl','training_525.pkl','training_530.pkl',
                   'training_535.pkl','training_540.pkl','training_545.pkl',
                   'training_550.pkl','training_555.pkl','training_560.pkl',
                   'training_565.pkl','training_570.pkl','training_575.pkl',
                   'training_580.pkl','training_585.pkl','training_590.pkl',
                   'training_595.pkl','training_600.pkl','training_605.pkl',
                   'training_610.pkl','training_615.pkl','training_620.pkl',
                   'training_625.pkl','training_630.pkl','training_635.pkl',
                   'training_640.pkl','training_645.pkl','training_650.pkl',
                   
                    'training_730.pkl','training_735.pkl','training_740.pkl',
                    'training_745.pkl','training_750.pkl','training_755.pkl',
                    'training_760.pkl','training_765.pkl','training_770.pkl',
                    'training_775.pkl','training_780.pkl','training_785.pkl',
                    'training_790.pkl','training_795.pkl','training_800.pkl',                   
                    'training_805.pkl','training_810.pkl','training_815.pkl',
                    'training_820.pkl','training_825.pkl','training_830.pkl',
                    'training_835.pkl', 'training_839.pkl',
                    'training_655.pkl','training_660.pkl',
                ###############   
                    'training_665.pkl',
                    'training_65.pkl',
                    'training_70.pkl','training_90.pkl','training_80.pkl',
                    'training_85.pkl','training_75.pkl','training_95.pkl',                   
                    'training_670.pkl','training_675.pkl','training_680.pkl',
                    'training_685.pkl','training_690.pkl','training_695.pkl',
                    'training_700.pkl','training_705.pkl','training_710.pkl',
                    'training_715.pkl','training_720.pkl','training_725.pkl',

                   ]


mld = MLDataPrep(isAnal=False)

mdp = MLDataProcess(model_name=None, isAnal=True)
####################
# mdp.load_model('./training_model/cnn_lstm_model_index.h5')
# mdp.model_name = './training_model/cnn_lstm_model_base.h5'
####################
mdp.load_model('./training_model/cnn_lstm_model_base.h5')
####################

filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
filenames.sort()
for file in filenames:
    if file in file_record:
        continue
    
    print(file)
    
    x_train, x_test, y_train, y_test = mld.prepare_stock_data_cnn(['{0}/{1}'.format(data_dir,file)])
    x_train = np.expand_dims(x_train, axis=2) 
    x_test = np.expand_dims(x_test, axis=2) 
 
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)
    mdp.process_model(mdp.model, x_train, x_test, y_train, y_test, epochs=3, batch_size=5, verbose=1)
      
    file_record.append(file)
    dump(file_record, open(record_file_path, 'wb'))
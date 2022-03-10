# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:02:50 2021

@author: Administrator
"""
# import dtl
import pandas as pd
from tqdm import tqdm
from  neural_alpha.data_generation import StockSeries, StockPanel, RollingFeatureDataSet_V2
import dtl.src.loader 
import dtl.src.toolbox
import numpy as np 
import utils


whole_period_start, whole_period_end  = '2017-01-01' ,'2022-02-15'

feature_rolling_window = 60
masked_name = 'invalid_series'
_class = 1

label_name = ['label_' + str(i) for i in range(_class)]

trainstart, trainend = '2015-01-01', '2019-01-01'
valstart, valend = '2019-01-10', '2020-01-01'
teststart, testend = '2020-01-10', '2021-01-01'



data_fillna_method = 'ignore'
na_tolerant_percent = 0.1
label_fillna_method = 'drop'



batched_dir = 'bydate'
batch_size = 500
data_dir = 'date'

benchmark = '000905.SH'
dataloader = dtl.src.loader.DataLoader()
availableUniverse = dtl.src.toolbox.StockUniverse.getAvailableStockUniverse(dataloader,
                                                                    newDays = (63, 63, 42))
availableUniverse = availableUniverse[
    filter(
    lambda x: False if x.split('.')[1] == 'BJ' else True,  
    availableUniverse.columns)]

predict_len = 5
cross_section_label_std = True

n_jobs = 1


def alphanetExpansion(arr3d):
    'open', 'high', 'low', 'close', 'volume', 'turn', 'return', 'vwap'
    
    
    replace_na_id = [4, 5]
    for i in replace_na_id:
        arr3d[:, :, i][arr3d[:, :, i] == 0] = np.nan
    
    volume2low = arr3d[:, :, 4]/ arr3d[:, :, 2]
    vwap2high = arr3d[:, :, 7]/ arr3d[:, :, 1]
    low2high = arr3d[:, :, 2]/ arr3d[:, :, 1]
    vwap2close = arr3d[:, :, 7]/arr3d[:, :, 3]
    turn2close = arr3d[:, :, 5]/arr3d[:, :, 3]
    turn2open = arr3d[:, :, 5]/ arr3d[:, :, 0]
    
    lst = [volume2low, vwap2high, low2high, vwap2close, turn2close, turn2open]
    for num, i in enumerate(lst):
        lst[num] = i.reshape(i.shape[0], i.shape[1], 1)
    lst = np.concatenate(lst, axis = -1)
    
    arr = np.concatenate([arr3d, lst], axis = -1)
    
    replace_mean_id = [4, 5, 8, 12, 13]
    for i in replace_mean_id:
        temp = pd.DataFrame(arr[:, :, i])
        temp = temp.fillna(temp.mean(axis = 1))
        arr[:, :, i] = temp.values
        
    log_id = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]    
    for i in log_id:
        arr[:, :, i] = np.log(arr[:, :, i])
    
    if np.isnan(arr).sum() !=0 :
        raise ValueError('Nan deteced in feature engineering')
    
    return arr

feature_eng = alphanetExpansion

#%%
#end of parameter setting
#----------------------------------------------------------------------------#

close = dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_CLOSE')
volume = dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_VOLUME')
open_ = dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_OPEN')
high = dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_HIGH')
low = dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_LOW')
amt = dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_AMOUNT')
preclose = dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_PRECLOSE')
adj = dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_ADJFACTOR')
floatshares = dataloader.read('MIXED', 'calculated', 'CAPS', 'FLOAT_SHR')
suspend = dataloader.read('WINDDB', 'calculated', 'PVProcess', 'ISSUSPEND')


market_close = dataloader.getMarketInfo(field = 'S_DQ_CLOSE', 
                  market_index = benchmark)
future_benchmark_return = market_close.astype(float).pct_change(periods = predict_len).shift(-predict_len)

close = close[whole_period_start : whole_period_end]

open_, high, low, close, volume, amt, preclose, adj, suspend, floatshares = utils._align(
    open_, high, low, close, volume, amt, preclose, adj, suspend, floatshares
    )
vwap = (amt * 10)/ volume
vwap = vwap.replace(0, np.nan).fillna(close)
turn = volume/floatshares.astype(float)

preclose, open_, high, low, close, vwap = preclose * adj, open_ * adj, high * adj, low * adj, close * adj, vwap * adj

return_ = close/preclose - 1

names = ['open', 'high', 'low', 'close', 'volume', 'turn', 'return', 'vwap']
stock_list = open_.columns

final_factor = pd.concat([i.unstack() for i in [open_, high, 
                                                low, close, 
                                                volume, turn, 
                                                return_, vwap]], axis = 1)
final_factor.columns = names
final_factor[masked_name] = suspend.unstack().fillna(True).astype(int)
label = final_factor.groupby(level = 0)['close'].pct_change(
    predict_len).shift(-predict_len).unstack().T
label = (label.T - future_benchmark_return.loc[label.index]).T
final_factor[label_name[0]] = label.unstack()
features = len(names)
lst = []
for id_ in tqdm(final_factor.index.levels[0]):
    if id_.split('.')[1] == 'BJ':
        continue
    d = final_factor.loc[id_][names]
    label = final_factor.loc[id_][label_name]
    masked = final_factor.loc[id_][[masked_name]]
    lst.append(StockSeries(d, label, masked, id_))

#%%
panel = StockPanel(lst) 
dataset = RollingFeatureDataSet_V2(whole_period_start, whole_period_end, 
                                panel, feature_rolling_window,
                                stockSpace = availableUniverse,
                                cross_sec_label_std = cross_section_label_std
                                )
dataset.run(
        data_fillna_method = data_fillna_method,
        na_tolerant_percent = na_tolerant_percent,
        label_fillna_method = label_fillna_method,
        n_jobs = n_jobs)



traindataset = dataset.getTFStyleSlice(trainstart, trainend, 
                                        batch = True,
                                        feature_eng = feature_eng,
                                        batch_dir = batched_dir,
                                        batch_size = batch_size,
                                        data_dir = data_dir)

valdataset = dataset.getTFStyleSlice(valstart, valend, 
                                      batch = True,
                                      feature_eng = feature_eng,
                                      batch_dir = batched_dir,
                                      batch_size = batch_size,
                                      data_dir = data_dir)



testdataset = dataset.getTFStyleSlice(teststart, testend, 
                                      batch = True,
                                      total_init = True,
                                      feature_eng = feature_eng,
                                      batch_dir = batched_dir,
                                      batch_size = batch_size,
                                      data_dir = data_dir)


#%%

#finish dataset preparation
#----------------------------------------------------------------------------
# dataset = []
# import gc
# gc.collect()
from alphanet_model import AlphaNetV3
import tensorflow as tf
model = AlphaNetV3(recurrent_unit = 'LSTM', hidden_units = 40, dropout = 0.1, l2 = 0.5)
#

# model = getModelV3(
#                     input_shape = [feature_rolling_window, features],
#                     recurrent_unit = 'LSTM', 
#                     hidden_units = 40,
#                     dropout = 0.1,
#                     l2 = 0.5,
#                     )

from metrics import IC, rankIC
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau




name = ''.join(str(datetime.datetime.now())[11:].split('.')[0].split(':'))
model_name = 'alphanetV3_'+ name

tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
stop = EarlyStopping(monitor="val_loss", patience = 10, restore_best_weights = True)
learning_rate_decay = ReduceLROnPlateau(monitor="val_batchRankIc", factor=0.5, patience = 3, min_delta= 1e-5)

model.compile(optimizer = Adam(0.001), metrics=[tf.keras.metrics.RootMeanSquaredError(), IC(),
                        rankIC()])
model.fit(traindataset.cache(),
          validation_data=valdataset.cache(),
          callbacks = [learning_rate_decay, stop, tensorboard], 
          shuffle = False,
          epochs=2000)

model.save_weights('alphanetV3_' + name + '.h5')
pred = model.predict(testdataset)
pred = dataset.transformPrediction(pred, teststart, testend)
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:59:04 2022

@author: Administrator
"""
import neural_alpha.alphanet_model
import neural_alpha.data_generation
import neural_alpha.utils
import pandas as pd 
import numpy as np
import dtl.src.toolbox
from tqdm import tqdm
from itertools import chain
from functools import reduce
import os
import gc



class ModelBaseFactor(object):
    def _align(df1, df2, *dfs):
        dfs_all = [df for df in chain([df1, df2], dfs)]
        if any(len(df.shape) == 1 or 1 in df.shape for df in dfs_all):
            dims = 1
        else:
            dims = 2
        mut_date_range = sorted(reduce(lambda x,y: x.intersection(y), (df.index for df in dfs_all)))
        mut_codes = sorted(reduce(lambda x,y: x.intersection(y), (df.columns for df in dfs_all)))
        if dims == 2:
            dfs_all = [df.loc[mut_date_range, mut_codes] for df in dfs_all]
        elif dims == 1:
            dfs_all = [df.loc[mut_date_range, :] for df in dfs_all]
        return dfs_all
    
    pass

class AlphaNetFactor(ModelBaseFactor):
    predict_len = 5
    feature_rolling_window = 60
    feature_size = 14
    shape = (None, feature_rolling_window, feature_size)
    data_fillna_method = 'ignore'
    na_tolerant_percent = 0.1
    label_fillna_method = 'drop'
    benchmark = '000905.SH'
    cross_section_label_std = True
    n_jobs = 1
    batched_dir = 'bydate'
    batch_size = 500
    data_dir = 'date'
    _class = 1
    label_name = ['label_' + str(i) for i in range(_class)]
    
    
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
        naid = np.where(np.isnan(arr[:, :, replace_mean_id]))
        z = arr[:, :, replace_mean_id]
        z[naid] = np.broadcast_to(
            np.expand_dims(
                np.nanmean(
                   arr[:, :, replace_mean_id], axis = 1), axis = 1), 
            arr[:, :, replace_mean_id].shape)[naid]
        
        arr[:, :, replace_mean_id] = z
        
            
        log_id = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]    
        for i in log_id:
            arr[:, :, i] = np.log(arr[:, :, i])
        
        if np.isnan(arr).sum() !=0 :
            raise ValueError('Nan deteced in feature engineering')
        
        return arr

    feature_eng = alphanetExpansion
    
    
    
    
    def __init__(self, model_hdf5_path, 
                 dataloader):
        self.model_hdf5_path = model_hdf5_path
        self.dataloader = dataloader
    
    
    def setUniverse(self):
        availableUniverse = dtl.src.toolbox.StockUniverse.getAvailableStockUniverse(
            self.dataloader,
            newDays = (63, 63, 42))
        self.availableUniverse = availableUniverse[
            filter(
            lambda x: False if x.split('.')[1] == 'BJ' else True,  
            availableUniverse.columns)]
    
    def setStockPanel(self, test = True):


        close = self.dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_CLOSE')
        volume = self.dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_VOLUME')
        open_ = self.dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_OPEN')
        high = self.dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_HIGH')
        low = self.dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_LOW')
        amt = self.dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_AMOUNT')
        preclose = self.dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_PRECLOSE')
        adj = self.dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_ADJFACTOR')
        floatshares = self.dataloader.read('MIXED', 'calculated', 'CAPS', 'FLOAT_SHR')
        suspend = self.dataloader.read('WINDDB', 'calculated', 'PVProcess', 'ISSUSPEND')
        
        
        close = close[self.whole_period_start : self.whole_period_end]
        
        open_, high, low, close, volume, amt, preclose, adj, suspend, floatshares = ModelBaseFactor._align(
            open_, high, low, close, volume, amt, preclose, adj, suspend, floatshares
            )
        vwap = (amt * 10)/ volume
        vwap = vwap.replace(0, np.nan).fillna(close)
        turn = volume/floatshares.astype(float)
        
        preclose, open_, high, low, close, vwap = preclose * adj, open_ * adj, high * adj, low * adj, close * adj, vwap * adj
        
        return_ = close/preclose - 1
        
        names = ['open', 'high', 'low', 'close', 'volume', 'turn', 'return', 'vwap']
        
        final_factor = pd.concat([i.unstack() for i in [open_, high, 
                                                        low, close, 
                                                        volume, turn, 
                                                        return_, vwap]], axis = 1)
        final_factor.columns = names
        final_factor['invalid_series'] = suspend.unstack().fillna(True).astype(int)
        
        if test:
            final_factor[self.label_name] = 0
        else:
            
            market_close = self.dataloader.getMarketInfo(field = 'S_DQ_CLOSE', 
                              market_index = self.benchmark)
            future_benchmark_return = market_close.astype(float).pct_change(
                periods = self.predict_len).shift(-self.predict_len)

            label = final_factor.groupby(level = 0)['close'].pct_change(
                self.predict_len).shift(-self.predict_len).unstack().T
            label = (label.T - future_benchmark_return.loc[label.index]).T
            final_factor[self.label_name] = pd.DataFrame(label.unstack())
            
            
        lst = []
        for id_ in tqdm(final_factor.index.levels[0]):
            if id_.split('.')[1] == 'BJ':
                continue
            d = final_factor.loc[id_][names]
            label = final_factor.loc[id_][self.label_name]
            masked = final_factor.loc[id_][['invalid_series']]
            lst.append(neural_alpha.data_generation.StockSeries(d, label, masked, id_))
        self.panel = neural_alpha.data_generation.StockPanel(lst) 
    
    
    
    def getTrainVal(self, traintime, valtime):
        trainstart, trainend = traintime
        valstart, valend = valtime
        dataset = neural_alpha.data_generation.RollingFeatureDataSet_V2(self.whole_period_start, 
                                                                        self.whole_period_end, 
                                        self.panel, self.feature_rolling_window,
                                        stockSpace = self.availableUniverse,
                                        cross_sec_label_std = self.cross_section_label_std
                                        )
        
        dataset.run(
                data_fillna_method = self.data_fillna_method,
                na_tolerant_percent = self.na_tolerant_percent,
                label_fillna_method = self.label_fillna_method,
                n_jobs = self.n_jobs)


        
        
        traindataset = dataset.getTFStyleSlice(trainstart, trainend, 
                                                batch = True,
                                                feature_eng = AlphaNetFactor.feature_eng,
                                                batch_dir = self.batched_dir,
                                                batch_size = self.batch_size,
                                                data_dir = self.data_dir)

        valdataset = dataset.getTFStyleSlice(valstart, valend, 
                                              batch = True,
                                              feature_eng = AlphaNetFactor.feature_eng,
                                              batch_dir = self.batched_dir,
                                              batch_size = self.batch_size,
                                              data_dir = self.data_dir)
        return traindataset, valdataset
        
    
    def trainModel(self, traindataset, valdataset):
        from neural_alpha.alphanet_model import AlphaNetV3
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

        from neural_alpha.metrics import IC, rankIC
        from tensorflow.keras.callbacks import TensorBoard
        import datetime
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau




        name = ''.join(str(datetime.datetime.now())[11:].split('.')[0].split(':'))
        model_name = 'alphanetV3_'+ name

        tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
        stop = EarlyStopping(monitor="val_loss", patience = 10, restore_best_weights = True)
        learning_rate_decay = ReduceLROnPlateau(monitor="val_batchRankIc", 
                                                factor=0.5, patience = 3, min_delta= 1e-5)

        model.compile(optimizer = Adam(0.001), metrics=[tf.keras.metrics.RootMeanSquaredError(), IC(),
                                rankIC()])
        model.fit(traindataset.cache(),
                  validation_data=valdataset.cache(),
                  callbacks = [learning_rate_decay, stop, tensorboard], 
                  shuffle = False,
                  epochs=2000)

        model.save_weights(os.path.join(self.model_hdf5_path ,'alphanetV3_' + name + '.h5'))

    
    
    
    def getPrediction(self):
        
        dataset = neural_alpha.data_generation.RollingFeatureDataSet_V2(self.whole_period_start, self.whole_period_end, 
                                        self.panel, self.feature_rolling_window,
                                        stockSpace = self.availableUniverse,
                                        cross_sec_label_std = self.cross_section_label_std
                                        )
        dataset.run(
                data_fillna_method = self.data_fillna_method,
                na_tolerant_percent = self.na_tolerant_percent,
                label_fillna_method = self.label_fillna_method,
                n_jobs = self.n_jobs)


        testdataset = dataset.getTFStyleSlice(self.teststart, self.testend, 
                                              batch = True,
                                              total_init = True,
                                              feature_eng = AlphaNetFactor.feature_eng,
                                              batch_dir = 'bydate',
                                              batch_size = 500,
                                              data_dir = 'date')
        pred = self.model.predict(testdataset)
        pred = dataset.transformPrediction(pred, self.teststart, self.testend)
        
        return pred
    
    
    
    def loadModel(self):
        self.model = neural_alpha.alphanet_model.AlphaNetV3(recurrent_unit = 'LSTM', 
                                               hidden_units = 40, dropout = 0.1, l2 = 0.5)
        self.model.build(self.shape)
        self.model.load_weights(self.model_hdf5_path)
    
    
    def setDays(self, time):
        now = str(pd.to_datetime(time))[:10]
        days = self.dataloader.getTradeDays()
        t = list(days).index(pd.to_datetime(time))
        whole_period_start, whole_period_end  = str(days[t - self.feature_rolling_window + 1])[:10] , now
        self.teststart, self.testend =  whole_period_start, whole_period_end
        self.whole_period_start, self.whole_period_end = whole_period_start, whole_period_end
    
    def setTestDays(self, teststart, testend):
        days = self.dataloader.getTradeDays()
        teststart = pd.Series(0, index = days)[teststart:].index[0]
        t = list(days).index(pd.to_datetime(teststart))
        whole_period_start, whole_period_end  = str(
            days[t - self.feature_rolling_window + 1])[:10] , testend

        self.teststart, self.testend =  whole_period_start, whole_period_end
        self.whole_period_start, self.whole_period_end = whole_period_start, whole_period_end
        
    
    def predict_one(self, time):
        self.loadModel()
        self.setDays(time)
        self.setUniverse()
        self.setStockPanel()
        return self.getPrediction()
    
    def train(self, 
              wholetime,
              traintime, valtime
              ):
        self.whole_period_start, self.whole_period_end = wholetime
        self.setUniverse()
        self.setStockPanel(False)
        traindataset, valdataset = self.getTrainVal(traintime, valtime)
        self.trainModel(traindataset, valdataset)
        
        
    def predict(self, testtime):
        teststart, testend = testtime
        self.loadModel()
        self.setTestDays(teststart, testend)
        self.setUniverse()
        self.setStockPanel()
        return self.getPrediction()

class AlphaNetFactorV2(AlphaNetFactor):
    
    def setUniverse(self):
        availableUniverse = dtl.src.toolbox.StockUniverse.getAvailableStockUniverseWithoutSPZDT(
            self.dataloader,
            newDays = (63, 63, 42))
        self.availableUniverse = availableUniverse[
            filter(
            lambda x: False if x.split('.')[1] == 'BJ' else True,  
            availableUniverse.columns)]
    
class AlphaNetFactor_MinuteFactor(AlphaNetFactorV2):
    feature_rolling_window = 5
    na_tolerant_percent = 0
    feature_size = 16
    shape = (None, feature_rolling_window * 16, feature_size)

    
    @staticmethod
    def alphanetExpansion(arr3d):
        'open', 'high', 'low', 'close', 'volume', 'turn', 'return', 'vwap', 'cjbs'
        
        
        replace_na_id = [4, 5, 8]
        for i in replace_na_id:
            arr3d[:, :, i][arr3d[:, :, i] == 0] = np.nan
        
        volume2low = arr3d[:, :, 4]/ arr3d[:, :, 2]
        vwap2high = arr3d[:, :, 7]/ arr3d[:, :, 1]
        low2high = arr3d[:, :, 2]/ arr3d[:, :, 1]
        vwap2close = arr3d[:, :, 7]/arr3d[:, :, 3]
        turn2close = arr3d[:, :, 5]/arr3d[:, :, 3]
        turn2open = arr3d[:, :, 5]/ arr3d[:, :, 0]
        vol2cjbs = arr3d[:, :, 4]/arr3d[:, :, 8]
        
        
        lst = [volume2low, vwap2high, low2high, vwap2close, turn2close, turn2open, vol2cjbs]
        for num, i in enumerate(lst):
            lst[num] = i.reshape(i.shape[0], i.shape[1], 1)
        lst = np.concatenate(lst, axis = -1)
        
        arr = np.concatenate([arr3d, lst], axis = -1)
        
        replace_mean_id = [4, 5, 8, 9, 13, 14, 15]
        naid = np.where(np.isnan(arr[:, :, replace_mean_id]))
        z = arr[:, :, replace_mean_id]
        z[naid] = np.broadcast_to(
            np.expand_dims(
                np.nanmean(
                   arr[:, :, replace_mean_id], axis = 1), axis = 1), 
            arr[:, :, replace_mean_id].shape)[naid]
        
        arr[:, :, replace_mean_id] = z
        
            
        log_id = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]    
        for i in log_id:
            arr[:, :, i] = np.log(arr[:, :, i])
        
        if np.isnan(arr).sum() !=0 :
            raise ValueError('Nan deteced in feature engineering')
        
        return arr

    feature_eng = alphanetExpansion
    
    
    
   
    def setStockPanel(self, test = True):
        
        Q_start = pd.to_datetime(self.whole_period_start).quarter
        Y_start = pd.to_datetime(self.whole_period_start).year
        Q_end = pd.to_datetime(self.whole_period_end).quarter 
        Y_end = pd.to_datetime(self.whole_period_end).year        
        
        
        df = pd.concat(self.dataloader.getQuarterMinData(Y_start, Q_start, Y_end, Q_end))
        df = df[self.whole_period_start : self.whole_period_end]
        data = neural_alpha.utils.addFeatures(df, self.dataloader)
        data['turn'] = data['vol']/data['shares']
        data['return'] = data['close']/data['yclose'] - 1
        data[['open', 'close', 'high', 'low', 'vwap']] = data[['open', 'close', 'high', 'low', 'vwap']].values * data[['adj']].values
        data = data[['open', 'high', 'low', 'close', 'vol', 'turn', 'return', 'vwap', 'cjbs']]     
        label = neural_alpha.utils.getLabel(self.dataloader, self.benchmark,
                     excess = True, predict_len = self.predict_len)
        mask = neural_alpha.utils.getMasked(self.dataloader)
        self.panel = neural_alpha.utils.getStockPanel(data, label ,mask, 
                                                      label_name = self.label_name,
                                                      test = test)
    
    
    def getTrainVal(self, traintime, valtime):
        trainstart, trainend = traintime
        valstart, valend = valtime
        dataset = neural_alpha.data_generation.RollingFeatureDataSet_V2_withSkipping(self.whole_period_start, 
                                                                        self.whole_period_end, 
                                        self.panel, self.feature_rolling_window,
                                        stockSpace = self.availableUniverse,
                                        cross_sec_label_std = self.cross_section_label_std
                                        )
        
        dataset.run(
                data_fillna_method = self.data_fillna_method,
                na_tolerant_percent = self.na_tolerant_percent,
                label_fillna_method = self.label_fillna_method,
                n_jobs = self.n_jobs)


        
        
        traindataset = dataset.getTFStyleSlice(trainstart, trainend, 
                                                batch = True,
                                                feature_eng = self.feature_eng,
                                                batch_dir = self.batched_dir,
                                                batch_size = self.batch_size,
                                                data_dir = self.data_dir)

        valdataset = dataset.getTFStyleSlice(valstart, valend, 
                                              batch = True,
                                              feature_eng = self.feature_eng,
                                              batch_dir = self.batched_dir,
                                              batch_size = self.batch_size,
                                              data_dir = self.data_dir)
        return traindataset, valdataset
        
    
    def getPrediction(self):
        
        dataset = neural_alpha.data_generation.RollingFeatureDataSet_V2_withSkipping(self.whole_period_start, self.whole_period_end, 
                                        self.panel, self.feature_rolling_window,
                                        stockSpace = self.availableUniverse,
                                        cross_sec_label_std = self.cross_section_label_std
                                        )
        dataset.run(
                data_fillna_method = self.data_fillna_method,
                na_tolerant_percent = self.na_tolerant_percent,
                label_fillna_method = self.label_fillna_method,
                n_jobs = self.n_jobs)


        testdataset = dataset.getTFStyleSlice(self.teststart, self.testend, 
                                              batch = True,
                                              total_init = True,
                                              feature_eng = self.feature_eng,
                                              batch_dir = 'bydate',
                                              batch_size = 500,
                                              data_dir = 'date')
        pred = self.model.predict(testdataset)
        pred = dataset.transformPrediction(pred, self.teststart, self.testend)
        
        return pred
    
    def trainModel(self, traindataset, valdataset):
        from neural_alpha.alphanet_model import AlphaNetV3
        import tensorflow as tf
        model = AlphaNetV3(recurrent_unit = 'LSTM', hidden_units = 40, 
                           dropout = 0.1, l2 = 0.5)
        #

        # model = getModelV3(
        #                     input_shape = [feature_rolling_window, features],
        #                     recurrent_unit = 'LSTM', 
        #                     hidden_units = 40,
        #                     dropout = 0.1,
        #                     l2 = 0.5,
        #                     )

        from neural_alpha.metrics import IC, rankIC
        from tensorflow.keras.callbacks import TensorBoard
        import datetime
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau




        name = ''.join(str(datetime.datetime.now())[11:].split('.')[0].split(':'))
        model_name = 'alphanetV3_'+ name

        tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
        stop = EarlyStopping(monitor="val_batchIc", patience = 20, restore_best_weights = True)
        learning_rate_decay = ReduceLROnPlateau(monitor="val_batchIc", 
                                                factor=0.5, patience = 3, min_delta= 1e-5)

        model.compile(optimizer = Adam(0.001), metrics=[tf.keras.metrics.RootMeanSquaredError(), IC(),
                                rankIC()])
        model.fit(traindataset.cache(),
                  validation_data=valdataset.cache(),
                  callbacks = [learning_rate_decay, stop, tensorboard], 
                  shuffle = False,
                  epochs=2000)

        model.save_weights(os.path.join(self.model_hdf5_path ,'alphanetV3_' + name + '.h5'))
        del model 
        tf.keras.backend.clear_session() 
        gc.collect()



class AlphaNetFactor_MinuteFactorV2(AlphaNetFactor_MinuteFactor):
    @staticmethod
    def alphanetExpansion(arr3d):

        arr = arr3d.copy()
        arr[:,:,6] = arr[:,:,6] + 1
        naid = np.where(arr == 0)
        arr[naid] = np.broadcast_to(
            np.expand_dims(
                np.nanmean(
                   arr, axis = 1), axis = 1), 
            arr.shape)[naid]
        
        if np.isnan(arr).sum() !=0 :
            raise ValueError('Nan deteced in feature engineering')
        
        return arr

    feature_eng = alphanetExpansion
    def trainModel(self, traindataset, valdataset):
        from neural_alpha.mymodel import getModelV3
        import tensorflow as tf
        # model = AlphaNetV3(recurrent_unit = 'LSTM', hidden_units = 40, 
        #                    dropout = 0.1, l2 = 0.5)
        
        shape = list(eval(str(traindataset).split(
            'shapes: ')[-1].split(
            ', types')[0].split(
                ', (')[0][1:]))[-2:]
        model = getModelV3( cross_ignore_num = [4],
                            input_shape = shape,
                            recurrent_unit = 'LSTM', 
                            hidden_units = 40,
                            dropout = 0.1,
                            l2 = 0.5,
                            )

        from neural_alpha.metrics import IC, rankIC
        from tensorflow.keras.callbacks import TensorBoard
        import datetime
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau




        name = ''.join(str(datetime.datetime.now())[11:].split('.')[0].split(':'))
        model_name = 'alphanetV3_'+ name

        tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
        stop = EarlyStopping(monitor="val_batchIc", patience = 20, restore_best_weights = True)
        learning_rate_decay = ReduceLROnPlateau(monitor="val_batchIc", 
                                                factor=0.5, patience = 3, min_delta= 1e-5)

        model.compile(loss = 'MSE',
            optimizer = Adam(0.001), metrics=[tf.keras.metrics.RootMeanSquaredError(), IC(),
                                rankIC()])
        model.fit(traindataset.cache(),
                  validation_data=valdataset.cache(),
                  callbacks = [learning_rate_decay, stop, tensorboard], 
                  shuffle = False,
                  epochs=2000)

        model.save_weights(os.path.join(self.model_hdf5_path ,'alphanetV3_' + name + '.h5'))
        del model 
        tf.keras.backend.clear_session() 
        gc.collect()
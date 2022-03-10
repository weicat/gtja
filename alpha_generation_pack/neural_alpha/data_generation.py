# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 09:14:40 2021

@author: Administrator
"""

import numpy as np 
import numba
from numba import njit
from functools import reduce
import joblib
import tensorflow as tf
import pandas as pd 
from copy import deepcopy
from tqdm import tqdm




def conventionFill(df,
                   fill_columns = 'close',
                   fill_zero_columns = ['volume', 'amt'],
                   fill_one_columns = ['low2high']
                   ):
    
    df[fill_columns] = df[fill_columns].fillna(method = 'ffill')
    for i in df.columns:
        if i in fill_zero_columns:
            df[i] = df[i].fillna(0)
        elif i in fill_one_columns:
            df[i] = df[i].fillna(1)
        else:
            df[i] = df[i].fillna(df[fill_columns])
        
    
    return df

@njit
def getRollingWindowFeature(init_data, init_label, 
                            data, label, 
                            window, 
                            valid_index, mapped_index):
 
    for i, j in zip(valid_index, mapped_index):
        if j < window - 1:
            continue
        
        init_data[i,:,:]  = data[j - window + 1 : j + 1, : ]
        init_label[i, :] = label[j, :]
    
    return init_data, init_label


@njit
def getRollingWindowFeatureMiniFeature(init_data, init_label, 
                            data, label, 
                            window, 
                            valid_index, mapped_index):
 
    for i, j in zip(valid_index, mapped_index):
        if j < window - 1:
            continue
        
        init_data[i,:,:]  = np.ascontiguousarray(data[j - window + 1 : j + 1, : ,:]).reshape(-1, data.shape[-1])
        init_label[i, :] = label[j, :]
    
    return init_data, init_label



class StockSeries(object):
    
    def __init__(self, data, label, masked_valid_series, stockid):
        self.all_data = data
        self.all_label = label
        self.all_masked_valid_series = masked_valid_series
        self.stockid = stockid
        
        self.data = self.all_data
        self.label = self.all_label
        self.masked_valid_series = self.all_masked_valid_series
        
        
    def getPeriod(self, start, end):
        self.data = self.all_data[start: end]
        self.label = self.all_label[start: end]
        self.masked_valid_series = self.all_masked_valid_series[start: end]
    
    #PriceVolume Data
    def getRollingFeatures(
            self,
            all_timeline_c,
            rolling_window = 30,
            data_fillna_method = 'convention',
            na_tolerant_percent = 0,
            label_fillna_method = 'drop',
            fill_columns = 'close',
            fill_zero_columns = ['volume', 'amt'],
            fill_one_columns = ['low2high']
            ):
        
            
            all_timeline = deepcopy(all_timeline_c)
        
            data = self.data
            label = self.label
            masked_valid_series = self.masked_valid_series
            
            
            data_arr = np.zeros(shape = (len(all_timeline), rolling_window, data.shape[1] ))
            data_arr [:] = np.nan
            
            label_arr = np.zeros(shape = (len(all_timeline), label.shape[1]))
            label_arr[:] = np.nan
            
            
            
        
            
            all_timeline.loc[data.index, data.columns] = data
            all_timeline.loc[label.index, label.columns] = label
            all_timeline.loc[masked_valid_series.index, masked_valid_series.columns] = masked_valid_series
            all_timeline[masked_valid_series.columns] = all_timeline[masked_valid_series.columns].fillna(1)
            
            
            
            masked_rolling = all_timeline[masked_valid_series.columns].rolling(window = rolling_window).sum().fillna(
                    rolling_window + 1)
            valid_index = np.where((masked_rolling/ rolling_window) <= na_tolerant_percent )[0]
            
            if label_fillna_method == 'drop':
                valid_label_index = all_timeline[label.columns].isna().any(axis = 1)
                valid_label_index = np.where(valid_label_index == False )[0]
                valid_index = set(valid_label_index).intersection(set(valid_index))
            
            
            if data_fillna_method == 'convention':   
                all_timeline.loc[all_timeline[masked_valid_series.columns[0]] == 1, data.columns] = np.nan
                filled_data = conventionFill(all_timeline.loc[:, data.columns],
                                              fill_columns = fill_columns,
                                              fill_zero_columns = fill_zero_columns,
                                              fill_one_columns = fill_one_columns)
                
                filled_label = all_timeline.loc[:, label.columns]
                mapped_index = valid_index
                
            elif data_fillna_method == 'ignore':
                filled_data = all_timeline.loc[:,data.columns]
                filled_label = all_timeline.loc[:, label.columns]
        
                new_valid_index = np.where(filled_data.isna().any(axis = 1) == False)[0]
                valid_index = set(valid_index).intersection(set(new_valid_index))
                mapped_index = valid_index
                
            elif data_fillna_method == 'fillprevious':
                all_timeline.loc[all_timeline[masked_valid_series.columns[0]] == 1, data.columns] = np.nan
                filled_data = all_timeline.loc[:, data.columns]
                temp = filled_data.isna().any(axis = 1)
                filled_data = filled_data.dropna()
                all_nonaindex = np.where(temp == False)[0]
                filled_label = all_timeline.loc[:, label.columns]
                filled_label = filled_label.iloc[all_nonaindex, :]
                
                mapped_index = list(map(lambda x: dict(zip(all_nonaindex, range(len(filled_data))))[x], valid_index))
                
            
            filled_data_arr = filled_data.values
            filled_label_arr = filled_label.values
            valid_index =  numba.typed.List(valid_index)
            mapped_index =  numba.typed.List(mapped_index)
            
            if len(valid_index) == 0:
                return (data_arr, label_arr)
            tuple_arr = getRollingWindowFeature(data_arr, label_arr,
                                              filled_data_arr, filled_label_arr,
                                              rolling_window, 
                                              valid_index, mapped_index)
        
            return tuple_arr 
    
    

class StockMinuteSeries(StockSeries):
    
    check_num = 16
    
    def append(self, series):
        if series.stockid != self.stockid:
            raise ValueError('not two same stock series')
        
        self.all_data = (self.all_data.append(series.all_data)).sort_index()
        self.all_label = (self.all_label.append(series.all_label)).sort_index()
        self.all_masked_valid_series = (self.all_masked_valid_series.append(
            series.all_masked_valid_series)).sort_index()
        
        
        self.all_data = self.all_data[~self.all_data.index.duplicated()]
        self.all_label = self.all_label[~self.all_label.index.duplicated()]
        self.all_masked_valid_series = self.all_masked_valid_series[
            ~self.all_masked_valid_series.index.duplicated()]

        
        self.data = self.all_data
        self.label = self.all_label
        self.masked_valid_series = self.all_masked_valid_series
    
    
    def getPeriod(self, start, end):
        self.data = self.all_data.loc[pd.to_datetime(start): pd.to_datetime(end)]
        self.label = self.all_label[start: end]
        self.masked_valid_series = self.all_masked_valid_series[start: end]
        
        if not len(self.data) % self.check_num == 0:
            raise ValueError('lose data')
        
                    
    #PriceVolume Data
    def getRollingFeatures(
            self,
            all_timeline_c,
            rolling_window = 5,
            data_fillna_method = 'convention',
            na_tolerant_percent = 0,
            label_fillna_method = 'drop',
            fill_columns = 'close',
            fill_zero_columns = ['volume', 'amt'],
            fill_one_columns = ['low2high']
            ):
            '''
            其他的方法没写好，默认全部drop，
            必须rollingwindow中 都有值才进入
            
            '''
            
            all_timeline = deepcopy(all_timeline_c)
        
            data = self.data
            label = self.label
            masked_valid_series = self.masked_valid_series
            
            
            data_arr = np.zeros(shape = (len(all_timeline), rolling_window * self.check_num, data.shape[1] ))
            data_arr [:] = np.nan
            
            label_arr = np.zeros(shape = (len(all_timeline), label.shape[1]))
            label_arr[:] = np.nan
            
           
            
            all_timeline.loc[label.index, label.columns] = label
            all_timeline.loc[masked_valid_series.index, masked_valid_series.columns] = masked_valid_series
            all_timeline[masked_valid_series.columns] = all_timeline[masked_valid_series.columns].fillna(1)
            all_timeline.loc[:, ['datavalid']] = 1
            all_timeline.loc[data.index.remove_unused_levels().levels[0],'datavalid'] = 0
            all_timeline[masked_valid_series.columns] = pd.DataFrame(1 - ( 1-  all_timeline[masked_valid_series.columns[0]]) *\
                (1 - all_timeline['datavalid']))
            
            
            masked_rolling = all_timeline[masked_valid_series.columns].rolling(window = rolling_window).sum().fillna(
                    rolling_window + 1)
            valid_index = np.where((masked_rolling/ rolling_window) <= na_tolerant_percent )[0]
            
            
            
            if label_fillna_method == 'drop':
                valid_label_index = all_timeline[label.columns].isna().any(axis = 1)
                valid_label_index = np.where(valid_label_index == False )[0]
                valid_index = set(valid_label_index).intersection(set(valid_index))
            
            
            if data_fillna_method == 'convention':   
                raise ValueError('UnImplemented')
                
            elif data_fillna_method == 'ignore':
                filled_data_arr = np.zeros(shape = (len(all_timeline), self.check_num, data.shape[1] ))
                filled_data_arr[:] = np.nan
                filled_label = all_timeline.loc[:, label.columns]
                rtime_index = all_timeline.index[list(valid_index)]
                mytemp_data_arr = data.loc[pd.to_datetime(rtime_index)].values
                
          
                mapped_index = valid_index
                
            elif data_fillna_method == 'fillprevious':
                raise ValueError('UnImplemented')
            
            
            if len(valid_index) == 0:
                return (data_arr, label_arr)
            
            if mytemp_data_arr.shape[0]/len(valid_index) != self.check_num:
                raise ValueError('CheckData minibar lost')
            
            filled_data_arr[list(valid_index), :, :] = mytemp_data_arr.reshape(-1, 
                                                                         self.check_num, 
                                                                         data.shape[1])   
            
            
            
            filled_data_arr = np.ascontiguousarray(filled_data_arr)
            filled_label_arr = filled_label.values
            valid_index =  numba.typed.List(valid_index)
            mapped_index =  numba.typed.List(mapped_index)
            

            tuple_arr = getRollingWindowFeatureMiniFeature(data_arr, label_arr,
                                              filled_data_arr, filled_label_arr,
                                              rolling_window, 
                                              valid_index, mapped_index)
        
            return tuple_arr 
    
    
    



class StockPanel(object):
    
    def __init__(self, time_series_lst, n_jobs = 4):
        
        self.time_series_lst = time_series_lst
        self.stock_num = len(self.time_series_lst)
        self.whole_time_lst = sorted(reduce(lambda x,y : x.union(y), [set(i.data.index) for i in self.time_series_lst]))
        self.time_columns = list(self.time_series_lst[0].data.columns) + \
                            list(self.time_series_lst[0].label.columns) + \
                            list(self.time_series_lst[0].masked_valid_series.columns)
        self.all_timeline = pd.DataFrame(np.nan, index = self.whole_time_lst, 
                                         columns = self.time_columns)
        self.n_jobs = n_jobs
    
    
    def __getitem__(self, name):
        if isinstance(name, int):
            return self.time_series_lst[name]
    
        elif isinstance(name, str):
            return self.time_series_lst[self.getStockID().index(name)]
    
    
        
        
    
    
    def getStockID(self):
        return [i.stockid for i in self.time_series_lst]
    
    def getPeriod(self, start, end):
        for i in self.time_series_lst:
            i.getPeriod(start, end)
        self.all_timeline = self.all_timeline[start: end]
    
    
    def getRollingFeatures(self, n_jobs = 4, 
                            **kargs):
        
        
        res = joblib.Parallel(n_jobs = n_jobs)(joblib.delayed(
                i.getRollingFeatures)(self.all_timeline, **kargs) for i in self.time_series_lst )
    
        return res
    

class StockMinutePanel(StockPanel):
    check_num = 16
    def __init__(self, time_series_lst, n_jobs = 4):
        
        self.time_series_lst = time_series_lst
        self.stock_num = len(self.time_series_lst)
        self.whole_time_lst = sorted(reduce(lambda x,y : x.union(y), [set(i.data.index.levels[0]) for i in self.time_series_lst]))
        self.time_columns = list(self.time_series_lst[0].label.columns) + \
                            list(self.time_series_lst[0].masked_valid_series.columns)
        self.all_timeline = pd.DataFrame(np.nan, index = self.whole_time_lst, 
                                         columns = self.time_columns)
        self.n_jobs = n_jobs
    
    def append(self, panel):
        stockid = self.getStockID()
        temp_lst = []
        for s in panel.time_series_lst:
            if s.stockid not in stockid:
                temp_lst.append(s)
            else:
                self.time_series_lst[stockid.index(s.stockid)].append(s)
            
        self.time_series_lst += temp_lst
    
        self.stock_num = len(self.time_series_lst)
        self.whole_time_lst = sorted(reduce(lambda x,y : x.union(y), [set(i.data.index.levels[0]) for i in self.time_series_lst]))
        self.time_columns = list(self.time_series_lst[0].label.columns) + \
                            list(self.time_series_lst[0].masked_valid_series.columns)
        self.all_timeline = pd.DataFrame(np.nan, index = self.whole_time_lst, 
                                         columns = self.time_columns)
        
        
    
    
    
    

class UnevenSequence(tf.keras.utils.Sequence):
      def __init__(self, x_batches, y_batches):
          # x_batches, y_batches are lists of uneven batches
          self.x, self.y = x_batches, y_batches
      def __len__(self):
          return len(self.x)
      def __getitem__(self, idx):
          batch_x = self.x[idx]
          batch_y = self.y[idx]
          return (batch_x, batch_y)



class RollingFeatureDataSet(object):
    
    
    def __init__(self, 
                  start, end,
                  panel,
                  rolling_window,                                    
                  stockSpace = None,
                  cross_sec_label_std = True
                  ):
        
        if isinstance(start, str):
            start = pd.to_datetime(start)
        
        if isinstance(end, str):
            end = pd.to_datetime(end)        
        
        self.start = start
        self.end = end

        self.cross_sec_label_std = cross_sec_label_std
        
        self.panel = panel
        self.panel.getPeriod(start, end)
        self.rolling_window = rolling_window
        self.stockSpace = stockSpace
    
    def getShape(self):
        
        xshape = (len(self.panel.time_series_lst),
                  len(self.panel.all_timeline),
                  self.rolling_window,
                  len(self.panel.time_series_lst[0].data.columns)
                  )
        
        yshape = (len(self.panel.time_series_lst),
                  len(self.panel.all_timeline),
                  len(self.panel.time_series_lst[0].label.columns)
                  )
        
        
        return xshape, yshape
    
    
    
    def run(self, **kargs):
        print('------------Initialization--------------')
        xshape, yshape = self.getShape()
        self.x = np.zeros(shape = xshape)
        self.y = np.zeros(shape = yshape)
        self.x[:] = np.nan
        self.y[:] = np.nan
        print('------------GetRollingFeatures-----------')
        res = self.panel.getRollingFeatures(rolling_window = self.rolling_window, **kargs)
        stock_nums = len(res)
        for num in range(stock_nums):
            trainx, trainy = res.pop(0)
            self.x[num, :, :, :] = trainx
            self.y[num, :, :] = trainy
        
        if self.cross_sec_label_std:
            self.y = self.y/np.nanstd(self.y, axis = 0)
        
        
        print('------------GetValidSamples-----------')
        self.getValidSamples()

    def getValidSamples(self):
        valid_temp = ~np.any(np.any(np.isnan(self.x), axis = -1), axis = -1)
        if self.stockSpace is None:
            valid = valid_temp
        
        else:
            nostocks = set(self.panel.getStockID()) - set(self.stockSpace.columns) 
            slicedStocks = self.stockSpace.loc[self.panel.all_timeline.index, :]
            if len(nostocks) != 0:
                slicedStocks[nostocks] = False
            slicedStocks = slicedStocks.loc[:, self.panel.getStockID()].T.values
            valid = slicedStocks * valid_temp
            
        self.valid = valid
        
    def getTFStyleSlice(self, 
                        time_start, time_end,
                        feature_eng = lambda x: x,
                        batch = True,
                        batch_size = 500,
                        batch_dir = 'bynum',
                        data_dir = 'stock'):
        
        chosenperiod = self.panel.all_timeline[time_start : time_end ]
        calibrated_time_start, calibrated_time_end = chosenperiod.index[0], chosenperiod.index[-1]
        time_start = list(self.panel.all_timeline.index).index(calibrated_time_start)
        time_end = list(self.panel.all_timeline.index).index(calibrated_time_end)
        
        temp_valid = deepcopy(self.valid)
        temp_valid[:, :time_start ] = False
        temp_valid[:, time_end + 1 :] = False        
        
        if not batch:
            batch_dir = 'bynum'
            
        if batch_dir == 'bynum':
            choosed = np.where(temp_valid )
            if data_dir == 'stock':
                pass
            elif data_dir == 'date':
                temp = sorted([(i,j) for i, j in zip(*choosed)], 
                                    key = lambda x: x[1])
                temp = np.array(temp)
                choosed = (np.array(temp)[:,0], np.array(temp)[:,1])
                
            x = tf.data.Dataset.from_tensor_slices(
                    (feature_eng(self.x[choosed]), self.y[choosed]) )
            if batch:
                return x.batch(batch_size)
            else:
                return x
        
        else:
            if batch_dir == 'bystock':
                x_lst = []
                y_lst = []
                for i in range(temp_valid.shape[0]):
                    spec = temp_valid[i, :]
                    choosed = np.where(spec)
                    if len(choosed[0]) == 0:
                        continue
                    x_lst.append(tf.constant(feature_eng(self.x[i,:,:,:][choosed])) )
                    y_lst.append(tf.constant(self.y[i,:, :][choosed]))
            
            elif batch_dir == 'bydate':
                x_lst = []
                y_lst = []
                for i in range(temp_valid.shape[1]):
                    spec = temp_valid[:, i]
                    choosed = np.where(spec)
                    if len(choosed[0]) == 0:
                        continue
                    x_lst.append(tf.constant(feature_eng(self.x[:,i,:,:][choosed])) )
                    y_lst.append(tf.constant(self.y[:,i, :][choosed]))                
        
            return UnevenSequence(x_lst, y_lst)
    
            
    

class RollingFeatureDataSet_V2(RollingFeatureDataSet):
    
    
    def getTempValidDataFrame(self, time_start, time_end):
        
        chosenperiod = self.panel.all_timeline[time_start : time_end ]
        calibrated_time_start, calibrated_time_end = chosenperiod.index[0], chosenperiod.index[-1]
        time_start = list(self.panel.all_timeline.index).index(calibrated_time_start)
        time_end = list(self.panel.all_timeline.index).index(calibrated_time_end)

        temp_valid = deepcopy(self.valid)
        temp_valid[:, :time_start ] = False
        temp_valid[:, time_end + 1 :] = False    
        return temp_valid
    
    
    def transformPrediction(self, prediction, time_start, time_end):
        
        temp_valid = self.getTempValidDataFrame(time_start, time_end)
        tup_lst = []
        for i in range(temp_valid.shape[1]):
            spec = temp_valid[:, i]
            choosed = np.where(spec)
            if len(choosed[0]) == 0:
                continue
            tup_lst.append((i, choosed))
            

        split_idx = np.cumsum([len(i[1][0]) for i in tup_lst])
        split_arr = np.split(prediction, split_idx[:-1])
        time = self.panel.all_timeline.index[[i[0] for i in tup_lst]]
        stocks = np.array(self.panel.getStockID())
        chosen_stocks = [stocks[i[1][0]] for i in tup_lst]


        lst = [pd.DataFrame(arr, index = stocks, columns = [tt]) for 
               stocks, tt, arr in zip(chosen_stocks, time, split_arr)]
        return pd.concat(lst, axis = 1).T
        
        
    
    def getTFStyleSlice(self, 
                        time_start, time_end,
                        feature_eng = lambda x: x,
                        raw= False,
                        batch = True,
                        batch_size = 500,
                        batch_dir = 'bynum',
                        data_dir = 'stock',
                        total_init = False):
        
        temp_valid = self.getTempValidDataFrame(time_start, time_end)
     
        
        
        
        if not batch:
            batch_dir = 'bynum'
        
        
        
        if batch_dir == 'bynum':
            choosed = np.where(temp_valid )
            if data_dir == 'stock':
                pass
            elif data_dir == 'date':
                temp = sorted([(i,j) for i, j in zip(*choosed)], 
                                    key = lambda x: x[1])
                temp = np.array(temp)
                choosed = (np.array(temp)[:,0], np.array(temp)[:,1])
                
            
            if raw:
                return feature_eng(self.x[choosed]), self.y[choosed]
            
            
            x = tf.data.Dataset.from_tensor_slices(
                    (feature_eng(self.x[choosed]), self.y[choosed]) )
            if batch:
                return x.batch(batch_size)
            else:
                return x
        
        else:
            if batch_dir == 'bystock':
                x_lst = []
                y_lst = []
                for i in range(temp_valid.shape[0]):
                    spec = temp_valid[i, :]
                    choosed = np.where(spec)
                    if len(choosed[0]) == 0:
                        continue
                    x_lst.append(feature_eng(self.x[i,:,:,:][choosed]))
                    y_lst.append(self.y[i,:, :][choosed])
            
            elif batch_dir == 'bydate':
                x_lst = []
                y_lst = []
                for i in tqdm(range(temp_valid.shape[1])):
                    spec = temp_valid[:, i]
                    choosed = np.where(spec)
                    if len(choosed[0]) == 0:
                        continue
                    x_lst.append(feature_eng(self.x[:,i,:,:][choosed]))
                    y_lst.append(self.y[:,i, :][choosed])                
            
            if raw:
                return x_lst, y_lst


            xshape = tf.TensorShape([None]).concatenate(x_lst[0].shape[1:])
            yshape = tf.TensorShape([None]).concatenate(y_lst[0].shape[1:])

        
                    
            if not total_init:
                def mytfGenerator():
                    for i in range(len(x_lst)):
                        yield x_lst.pop(0), y_lst.pop(0)
                    
            else:
                def mytfGenerator():
                    for x, y in zip(x_lst, y_lst):
                        yield x, y
                    
            signature = (tf.TensorSpec(shape = xshape, dtype = tf.float32, name = "x"),
                         tf.TensorSpec(shape = yshape, dtype = tf.float32, name = 'y'))

                
            x = tf.data.Dataset.from_generator(mytfGenerator, 
                                               output_signature = signature
                                                   )
            return x 


class RollingFeatureDataSet_V2_withSkipping(RollingFeatureDataSet_V2):
    
    def getShape(self):
        
        xshape = (len(self.panel.time_series_lst),
                  len(self.panel.all_timeline),
                  self.rolling_window * self.panel.check_num,
                  len(self.panel.time_series_lst[0].data.columns)
                  )
        
        yshape = (len(self.panel.time_series_lst),
                  len(self.panel.all_timeline),
                  len(self.panel.time_series_lst[0].label.columns)
                  )
        
        
        return xshape, yshape
    




class RollingFeatureDataSetClassification_V2(RollingFeatureDataSet):

    def getTempValidDataFrame(self, time_start, time_end):
        
        chosenperiod = self.panel.all_timeline[time_start : time_end ]
        calibrated_time_start, calibrated_time_end = chosenperiod.index[0], chosenperiod.index[-1]
        time_start = list(self.panel.all_timeline.index).index(calibrated_time_start)
        time_end = list(self.panel.all_timeline.index).index(calibrated_time_end)

        temp_valid = deepcopy(self.valid)
        temp_valid[:, :time_start ] = False
        temp_valid[:, time_end + 1 :] = False    
        return temp_valid
    
    def transformPrediction(self, prediction, time_start, time_end):
        
        temp_valid = self.getTempValidDataFrame(time_start, time_end)
        tup_lst = []
        for i in range(temp_valid.shape[1]):
            spec = temp_valid[:, i]
            choosed = np.where(spec)
            if len(choosed[0]) == 0:
                continue
            tup_lst.append((i, choosed))
            

        split_idx = np.cumsum([len(i[1][0]) for i in tup_lst])
        split_arr = np.split(prediction, split_idx[:-1])
        time = self.panel.all_timeline.index[[i[0] for i in tup_lst]]
        stocks = np.array(self.panel.getStockID())
        chosen_stocks = [stocks[i[1][0]] for i in tup_lst]


        lst = [pd.DataFrame(arr, index = stocks, columns = [tt]) for 
               stocks, tt, arr in zip(chosen_stocks, time, split_arr)]
        return pd.concat(lst, axis = 1).T
        
        
    
    def getTFStyleSlice(self, 
                        time_start, time_end,
                        feature_eng = lambda x: x,
                        raw = False,
                        drop_percent = 0,
                        batch = True,
                        batch_size = 500,
                        batch_dir = 'bynum',
                        data_dir = 'stock',
                        total_init = False):
        
        temp_valid = self.getTempValidDataFrame(time_start, time_end)
        split_percent = ((1- drop_percent)/2, (1 + drop_percent)/2)
        res_down = self.y < np.nanquantile(self.y, split_percent[0], axis = 0)
        res_up = self.y >= np.nanquantile(self.y, split_percent[1], axis = 0)
        mylabel = np.concatenate([res_up, res_down], axis = -1)
        label_valid = np.any(mylabel, axis = -1)        
        temp_valid = temp_valid & label_valid
            
    
        
        if not batch:
            batch_dir = 'bynum'
            
        if batch_dir == 'bynum':
            choosed = np.where(temp_valid)
            if data_dir == 'stock':
                pass
            elif data_dir == 'date':
                temp = sorted([(i,j) for i, j in zip(*choosed)], 
                                    key = lambda x: x[1])
                temp = np.array(temp)
                choosed = (np.array(temp)[:,0], np.array(temp)[:,1])
                
            if raw:
                return feature_eng(self.x[choosed]), mylabel[choosed], np.abs(self.y[choosed])
                
            
            
            x = tf.data.Dataset.from_tensor_slices(
                    (feature_eng(self.x[choosed]), mylabel[choosed], np.abs(self.y[choosed]) ) )
        
        
        
            if batch:
                return x.batch(batch_size)
            else:
                return x
        
        else:
            if batch_dir == 'bystock':
                x_lst = []
                y_lst = []
                z_lst = []
                for i in range(temp_valid.shape[0]):
                    spec = temp_valid[i, :]
                    choosed = np.where(spec)
                    if len(choosed[0]) == 0:
                        continue
                    x_lst.append(feature_eng(self.x[i,:,:,:][choosed]))
                    y_lst.append(mylabel[i, :, :][choosed])
                    z_lst.append(np.abs(self.y[i,:, :][choosed]))
            
            elif batch_dir == 'bydate':
                x_lst = []
                y_lst = []
                z_lst = []
                for i in tqdm(range(temp_valid.shape[1])):
                    spec = temp_valid[:, i]
                    choosed = np.where(spec)
                    if len(choosed[0]) == 0:
                        continue
                    x_lst.append(feature_eng(self.x[:,i,:,:][choosed]))
                    y_lst.append(mylabel[:, i, :][choosed])
                    z_lst.append(np.abs(self.y[:,i, :][choosed]))                
            
            if raw:
                return x_lst, y_lst, z_lst

            xshape = tf.TensorShape([None]).concatenate(x_lst[0].shape[1:])
            yshape = tf.TensorShape([None]).concatenate(y_lst[0].shape[1:])
            zshape = tf.TensorShape([None]).concatenate(z_lst[0].shape[1:])

            if not total_init:
                def mytfGenerator():
                    for i in range(len(x_lst)):
                        yield x_lst.pop(0), y_lst.pop(0), z_lst.pop(0)
                    
            else:
                def mytfGenerator():
                    for x, y, z in zip(x_lst, y_lst, z_lst):
                        yield x, y, z
                    
            signature = (tf.TensorSpec(shape = xshape, dtype = tf.float32, name = "x"),
                         tf.TensorSpec(shape = yshape, dtype = tf.float32, name = 'y'),
                         tf.TensorSpec(shape = zshape, dtype = tf.float32, name = 'z')
                         )

                
            x = tf.data.Dataset.from_generator(mytfGenerator, 
                                               output_signature = signature
                                                   )
            return x 

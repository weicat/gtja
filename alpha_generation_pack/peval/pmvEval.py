# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:58:09 2022

@author: lth
"""

import pandas as pd 
import numpy as np
from tqdm import tqdm
import utils.numba_backtest_utils 
import utils.pandas_utils                

class WeightPosition(object):
    '''
    这个fill可以想一下，是不是要全集、还是只要有权重的就可以
    '''

    
    def __init__(self, weight_position):
        if isinstance(weight_position, list):
            self.weight_position = pd.concat(weight_position).T
        elif isinstance(weight_position, pd.DataFrame):
            self.weight_position = weight_position
        
        else:
            raise ValueError('cannot read dollarposition')
        
        self.weight_position.index = pd.to_datetime(self.weight_position.index)
    def fill(self, stock, time):
        z = set(self.weight_position.columns) - set(stock)
        if len(z) > 0:
            print(f'{z} not in stocks, automatically delete')
            
        temp = self.weight_position[set(stock).intersection(
            set(self.weight_position.columns))]
        temp.loc[:, set(stock) - set(temp.columns)] = 0
        temp = temp.loc[:, stock]
        temp.index = np.array(time)[np.searchsorted(pd.to_datetime(time), 
                                     pd.to_datetime(temp.index) )]
        temp.index = pd.to_datetime(temp.index)
        self.weight_position = temp[~temp.index.duplicated(keep = 'last')]
        self.rebalance_index = np.searchsorted(pd.to_datetime(time),
                                               self.weight_position.index)
        

    @property
    def index(self):
        return self.weight_position.index
    
    @property
    def values(self):
        return self.weight_position.values

    @property
    def df(self):
        return self.weight_position


    class PositionIterator(object):
        def __init__(self,obj):
            self.obj = obj
            self.index = 0
    
        def __iter__(self):
            pass
    
        def __next__(self):
            if self.index < len(self.obj.weight_position.index):
                ret = self.obj.weight_position.index[self.index]
                self.index += 1
                return ret
            else:
                raise StopIteration
                
    def __getitem__(self, key):
        return self.weight_position.loc[pd.to_datetime(key)].dropna()
    
    
    def __iter__(self):
        return self.PositionIterator(self)
    
      
    def __len__(self):
        
        return len(self.weight_position) 


class WeightPositions(object):
    
    def __init__(self, *weight):
        self.weight_lst = weight
    
    def fill(self, stock, time):
        for i in self.weight_lst:
            i.fill(stock, time)
    
    
    @property
    def index(self):
        return self.weight_lst[0].index
    

    @property
    def df(self):
        return [i.df for i in self.weight_lst]
    
    @property
    def rebalance_index(self):
        return [i.rebalance_index for i in self.weight_lst]


class PortfolioResult(object):
    '''
    简单的factor 分组测试
    用的是weight/假设一天之内weight不变
    日/周 换仓没问题，月换仓有问题
    long only 
    
    
    注意：如果是position_lst, 需要多个portfolio时间一致
    '''
    
    
    def __init__(self, weight_position):
        self.weight_position = weight_position
        
    
    
    def __TEMPFUNC__(self, 
                     weight_arrs,
                     balance_index,
                     adjclose,
                     sts,
                     suspends,
                     zts,
                     dts):
    
        weight_arrs, adjclose, sts, suspends, zts, dts =  utils.pandas_utils.toValuesWithNone(
            weight_arrs, adjclose, sts, suspends, zts, dts)
        
    
        
        return_arr, real_volume_arr = utils.numba_backtest_utils.getPortfolioReturn(
            weight_arrs,
            balance_index,
            adjclose, 
            sts, suspends, zts, dts)
        
        
        
        
        return  pd.Series(return_arr, index = self.record_dates),\
                pd.DataFrame(real_volume_arr,
                                    index = self.record_dates,
                                    columns = self.record_stocks)     

    def getPortfolioMV(self,
                        adjclose,
                        remove_st = False,
                        remove_suspend = False,
                        remove_zt = False,
                        remove_dt = False):
        
        
        '''
        如果不要st/... 直接用非dataframe的任何东西都可以
        这里的判断逻辑是 remove_st/... isinstance pd.DataFrame
        '''

        
        sts, suspends, zts, dts = None, None, None, None
        '''
        这里填NA的逻辑就是退市的股票怎么处理
        照道理来说退市股票的量价数据为nan,应该处理起来没有多大区别
        所以都填0/1应该问题不大
        '''
        if isinstance(remove_st, pd.DataFrame):
            sts = remove_st
        if isinstance(remove_suspend, pd.DataFrame):
            suspends = remove_suspend
        if isinstance(remove_zt, pd.DataFrame):
            zts = remove_zt
        if isinstance(remove_dt, pd.DataFrame):
            dts = remove_dt

        start, end = self.weight_position.index[0], self.weight_position.index[-1]

        adjclose, sts, suspends, zts, dts =  utils.pandas_utils._sliceIndex(
            start, end, adjclose, sts, suspends, zts, dts)

            
        adjclose, sts, suspends, zts, dts =  utils.pandas_utils._alignWithNone(
            adjclose, sts, suspends, zts, dts)
        
        self.record_stocks, self.record_dates = adjclose.columns, adjclose.index
        self.weight_position.fill(self.record_stocks, self.record_dates)
        
        weight_arrs = self.weight_position.df
        balance_index = self.weight_position.rebalance_index
        
        if isinstance(weight_arrs, list):
            _mv = []
            _position = []
            for w, b in tqdm(zip(weight_arrs, balance_index), total = len(weight_arrs) ):
                mv, position = self.__TEMPFUNC__(w,
                                  b,
                                  adjclose,
                                  sts,
                                  suspends,
                                  zts,
                                  dts)         
                _mv.append(mv)
                _position.append(position)
            return _mv, _position
    
        else:
            return self.__TEMPFUNC__(weight_arrs,
                              balance_index,
                              adjclose,
                              sts,
                              suspends,
                              zts,
                              dts)    
                
        
        

        
            
        
        
        

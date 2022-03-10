# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:30:48 2022

@author: Administrator
"""
import pandas as pd 
from tqdm import tqdm 

class Backtest(object):
    
    pass


class BacktestWithAlpha(object):
    fake_mv = 100
    round_tc = 0.003
    def __init__(self, universe, optimizer, alpha_forecast, periods = 1):
        self.universe = universe
        self.alpha_forecast = alpha_forecast
        self.optimizer = optimizer
        self.loader = optimizer.dataloader
        self.benchmark_index = optimizer.benchmark_index
        self.periods = periods
        
        self.dollar_weight_lst = []
        self.tc = pd.Series()
    
    def recordTC(self, df1, df2):
        df = pd.concat([df1, df2],axis = 1).fillna(0)
        fee = (df.diff(axis = 1).iloc[:, -1]).abs().sum() * self.round_tc/2
        self.tc.loc[pd.to_datetime(self.time)] = fee
    def getNewDollarPosition(self):
        forecast = self.alpha_forecast.loc[self.time].dropna()
        chosen_stocks = self.universe.loc[self.time]
        self.forecast = forecast[set(forecast.index).intersection(set(chosen_stocks[chosen_stocks == True].index))]
        temp = self.optimizer.optimize(self.iterate, 
                                       self.dollar_position, 
                                       self.forecast, self.time)
        self.recordTC(temp, self.dollar_position)
        self.dollar_position = temp
        self.dollar_position = self.dollar_position[self.dollar_position != 0]
        
    def updateVolumePositionFromDollarPosition(self):
        close = self.loader.getAdjClose().loc[self.time, self.dollar_position.index]
        self.volume_position = self.fake_mv * self.dollar_position/close        
        
    def updateMarketValue(self):
        close = self.loader.getAdjClose().loc[self.time, 
                                              self.volume_position.index]
        self.mv = (close * self.volume_position).sum()    
    
    def updateReturn(self):
        self.ret = self.mv/self.fake_mv - 1
    
    def updateDollarPosition(self):
        if self.iterate  !=0:
            close = self.loader.getAdjClose().loc[self.time, 
                                                  self.volume_position.index]
            if close.isna().sum()!=0:
                print('{t} has problem'.format(t = self.time))
                print(close[close.isna()])
            self.dollar_position = (close * self.volume_position)/(close * self.volume_position).sum()   
        
    def setClock(self, iterate, time):
        self.iterate = iterate
        self.time = time
    
    def init(self):
        self.dollar_position = pd.Series(0, 
                                index = self.alpha_forecast.iloc[0,:].dropna().index)
        self.volume_position = pd.Series(0, 
                                index = self.alpha_forecast.iloc[0,:].dropna().index)
    
    def runOnePeriod(self, iterate, time):
        self.setClock(iterate, time)
        self.updateMarketValue()
        self.updateReturn()
        self.updateDollarPosition()
        if self.iterate % self.periods == 0:
            self.getNewDollarPosition()
        self.updateVolumePositionFromDollarPosition()
        self.dollar_weight_lst.append(self.dollar_position)        
        
    def run(self):
        self.init()
        portfolio_return = []
        for iterate, time in tqdm(enumerate(self.alpha_forecast.index),
                                  total = len(self.alpha_forecast),
                                  desc = '回测中'):
            self.runOnePeriod(iterate, time)
            if iterate !=0:
                portfolio_return.append(pd.Series(self.ret, index = [pd.to_datetime(time)], name = 'portfolio'))
        portfolio_return = pd.concat(portfolio_return)
        benchmark_return = self.loader.getMarketRet(self.benchmark_index)
        benchmark_return.index = pd.to_datetime(benchmark_return.index)
        benchmark_return = benchmark_return.loc[portfolio_return.index]
        return pd.concat([portfolio_return, benchmark_return], axis = 1)
        
        
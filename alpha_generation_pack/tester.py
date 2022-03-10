# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:19:46 2022

@author: Administrator
"""

import dtl.src.loader as dsl
import dtl.src.toolbox as dst
import pandas as pd 

dataloader = dsl.DataLoader()
barraloader = dsl.BarraLoader(database_path = r'Z:\LuTianhao\StaticBarra_CopyedFrom20220302')
universe = dst.StockUniverse.getAvailableStockUniverse(dataloader)
alphaforecast = pd.read_csv(r'H:/alpha_generation_pack/forecast1_min_nozdt.csv',index_col = 0)

#%%
from portfolio.backtest import BacktestWithAlpha
from portfolio.optimizer import BarraOptimizer

opt = BarraOptimizer(dataloader, barraloader)
opt.setOptParams({
      'risk_passive_relative_rest' : 0.2,
       # 'ind_passive_gross_rest' : 0.02,
     'ind_passive_relative_rest' : 0.2,
     # 'weight_passive' :0.025,
       'round_turnover': 0.2
})
tester = BacktestWithAlpha(universe, opt, alphaforecast['2021-01-01':], periods = 1)
res = tester.run()

cross = (res.iloc[:,0] - res.iloc[:,1])
adjusted = pd.concat([tester.tc, cross], axis = 1).fillna(0).diff(axis = 1).iloc[:,1]
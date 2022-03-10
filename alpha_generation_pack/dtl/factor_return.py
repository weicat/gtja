# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:27:36 2022

@author: Administrator
"""
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import src.loader
import src.toolbox 
import pandas as pd 


database_path = r'Z:\LuTianhao\DataBase'
factor_root_path = r'Z:\LuTianhao\Barra'


dataloader = src.loader.DataLoader(database_path = database_path)
barraloader = src.loader.BarraLoader(database_path = factor_root_path)
universe = src.toolbox.StockUniverse.getAvailableStockUniverse(dataloader ,
                                                       newDays = (252, 252, 63), 
                                                       suspendDays = 63, 
                                                       recoverDays = 5)
used_factors = ['Beta', 'BooktoPrice',
 'EarningsYield', 'Growth',
 'Leverage', 'Liquidity',
 'Momentum', 'NonLinearSize',
 'ResidualVolatility', 'Size']

solver = src.toolbox.SolveBarraToolBox(dataloader, 
                           barraloader,
                           used_factors,
                           universe,
                           )

ret = solver.getFactorRet()
import datetime
ret.to_pickle(r'Z:\LuTianhao\Barra\factor_return\factor_return_{d}.pickle'.format(
    d = str(pd.to_datetime(datetime.datetime.now()))[:10]
    ))
# 

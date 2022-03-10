# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:47:28 2021

@author: Administrator
"""
import pandas as pd
import numpy as np
import dtl.src.barra_mod as barra
import os
import dtl.src.new_utils as utils

class Factor(object):
    
    '''
    NAME 是自己的class的名字
    name 是输出因子的名字
    '''
    
    version = 'Barra_CNE5'
    fundamental = False
    name = []
    NAME = None
    
    def __init__(self, folder_path):
        self.folder_path = folder_path
        if len(self.name) == 0 :
            raise ValueError('NO output name')
        
        if self.NAME is None:
            raise ValueError('Factor name should be given')
        
    def cal(self):
        pass
    
    def update(self):
        pass
    
    def getName(self):
        return ['_'.join(self.version, name) for name in self.name]
    
    def save(self):
        
        for name in self.val.keys():
            self.val[name].index = pd.to_datetime(self.val[name].index)
            
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)
            self.val[name].to_pickle(os.path.join(self.folder_path, name + '.pickle'))
    
    
    def close(self):
        self.val = {}
    
    def load(self):
        
        # temp = [pd.read_pickle(os.path.join(filepath, name + '.pickle') ) for name in self.name]
        temp = []
        for name in self.name:
            df = pd.read_pickle(os.path.join(self.folder_path, name + '.pickle'))
            col = df.columns
            choosedcol = []
            for j in col:
                if j[:2] == 'NE':
                    continue
                choosedcol.append(j)
            temp.append(df[choosedcol])


        for i in temp:
            i.index = pd.to_datetime(i.index)
        
        self.val = dict(zip(self.name, temp))
    
    
        
    

class Size(Factor):
    NAME = 'SIZE'
    name = ['Size']
    
    def cal(self, dataloader):
        
        size = dataloader.getMarketValue()
        size = np.log(size)
        self.val = dict(zip(self.name, [size]))


class Beta(Factor):
    NAME = 'BETA'
    name = ['Beta', 'Hsigma']
    half_life = 63
    window = 252
        
    def cal(self, dataloader):
        
        stock_ret = dataloader.getStockRetBoundedAndDropFirstFive()
        market_ret = dataloader.getMarketRet()
        val = barra.beta(stock_ret, market_ret, self.half_life, self.window)
        self.val = dict(zip(self.name, val))
    
class RSTR(Factor):
    NAME = 'RSTR'
    name = ['RSTR']
    long_lag = 252
    short_lag = 21 
    half_life = 126
    riskfree = 0.03/252
        
    def cal(self, dataloader):
        
        
        stock_ret = dataloader.getStockRetBoundedAndDropFirstFive()
        val = barra.momentum(stock_ret,
                             longlag = self.long_lag,
                             half_life = self.half_life,
                             riskfree = self.riskfree)
        val = pd.DataFrame(val, index = stock_ret.index[self.long_lag -1:], 
                           columns = stock_ret.columns).shift(self.short_lag)
        self.val = dict(zip(self.name, [val]))

class DaStd(Factor):
    NAME = 'DASTD'
    name = ['DaStd']
    window = 252
    half_life = 42
    riskfree = 0.03/252 
        
    
    def cal(self, dataloader):
        
        stock_ret = dataloader.getStockRetBoundedAndDropFirstFive()
        val = barra.dastd(stock_ret,
                          window = self.window,
                          half_life = self.half_life,
                          riskfree = self.riskfree)
        
        val = pd.DataFrame(val, index = stock_ret.index[self.window - 1:], columns = stock_ret.columns)
        self.val = dict(zip(self.name, [val]))



class CMRA(Factor):
    NAME = 'CMRA'
    name = ['CMRA']
    month_len = 12
    month_days = 21
    riskfree= 0.03/252


    def cal(self, dataloader):
        
        r = dataloader.getStockRetBoundedAndDropFirstFive()
        # r = dataloader.getRawStockRet()
        val = barra.cmra(r, 
                         month_len = self.month_len,
                         month_days = self.month_days,
                         riskfree = self.riskfree
                         )
    
        self.val = dict(zip(self.name, [val]))
    

class NonLinearSize(Factor):
    NAME = 'NONLINEARSIZE'
    name = ['NonLinearSize']
    wls = False
    winsorize = 5
    def cal(self, dataloader):
        
        size = dataloader.getMarketValue()
        val = barra.nonlinearsize(
                                    size,
                                    wls = self.wls,
                                    )
        self.val = dict(zip(self.name, [val]))

class BooktoPrice(Factor):
    NAME = 'BOOKTOPRICE'
    name = ['BP']
    fundamental = True

    
    def cal(self, dataloader):
        
        TotalEquity = dataloader.getBookValue() 
        MarketValue = dataloader.getMarketValue()
        val = TotalEquity/MarketValue
        self.val = dict(zip(self.name, [val]))
    
    

class ETOP(Factor):
    NAME = 'ETOP'
    name = ['ETOP', 'CETOP']
    fundamental = True
    def cal(self, dataloader):
        
        MarketValue = dataloader.getMarketValue()
        
        netprofit = dataloader.getTrailingNetProfit()
        cashnetprofit = dataloader.getTrailingCashNetProfit()
        
        etop = barra.etop_func(netprofit)
        cetop = barra.etop_func(cashnetprofit)
        
        etop = dataloader.unstack(etop).fillna(method= 'ffill')
        cetop = dataloader.unstack(cetop).fillna(method = 'ffill')
        
        etop = etop/MarketValue
        cetop = cetop/MarketValue
        
        val = [etop, cetop]
        self.val = dict(zip(self.name, val))
        
    

class STO(Factor):
    NAME = 'STO'
    name = ['STOM', 'STOQ', 'STOA']

    def cal(self, dataloader):
        
        turnover = dataloader.getTurnOver()
        STOM, STOQ, STOA = barra.sto_func(turnover)
        self.val = dict(zip(self.name, [STOM, STOQ, STOA]))
    

class EGRF(Factor):
    NAME = 'EGRF'
    name = ['EGRLF', 'EGRSF']
    
    def cal(self, dataloader):
        
        netprofit = dataloader.getTrailingNetProfit()
        egrlf = barra.egrf_func(netprofit, 12)
        egrsf = barra.egrf_func(netprofit, 4)
        egrlf = dataloader.unstack(egrlf).fillna(method = 'ffill')
        egrsf = dataloader.unstack(egrsf).fillna(method = 'ffill')
        self.val = dict(zip(self.name, [egrlf, egrsf]))



class GRO(Factor):
    NAME = 'GRO'
    name = ['EGRO', 'SGRO']

        
    def cal(self, dataloader):
        
        netprofit = dataloader.getTrailingNetProfit()
        egro = barra.gro_func(netprofit)
        egro = dataloader.unstack(egro).fillna(method= 'ffill').replace('P', np.nan)
        
        operrev = dataloader.getTrailingOperRevenue()
        sgro = barra.gro_func(operrev)
        sgro = dataloader.unstack(sgro).fillna(method = 'ffill').replace('P', np.nan)

    
        self.val = dict(zip(self.name, [egro, sgro]))
        


class LEV(Factor):
    NAME = 'LEV'
    name = ['MLEV', 'BLEV']
    
    def cal(self, dataloader):
        MV = dataloader.getMarketValue().astype(float)
        LD = dataloader.getNonCurLiability().astype(float)
        PriorMV = dataloader.getPriorSharesMV().astype(float)
        PriorBond = dataloader.getPriorSharesDebt().astype(float)
        PE = PriorMV + PriorBond
        BE = dataloader.getBookValue().astype(float)
        DEBT = PE + LD
        mlev = 1 + DEBT/MV
        blev = 1 + DEBT/BE
        
                
        self.val = dict(zip(self.name, [mlev, blev]))


class DTOA(Factor):
    NAME = 'DTOA'
    name = ['DTOA']
    
    def cal(self, dataloader):
        
        TotalAsset = dataloader.getTotalAsset().astype(float)
        TotalLiability = dataloader.getTotalLiability().astype(float)
        dtoa = TotalLiability * 1/TotalAsset * 1
        self.val = dict(zip(self.name, [dtoa]))


class SecondaryFactor(Factor):
    version = 'Barra_CNE5_SecondaryFactor'
    USED_FACTOR_DICT = {}
    IS_STANDARDIZE = False
    FUNDAMENTAL = False
    
    def __init__(self, folder_path, *args):
        self.folder_path = folder_path
        if len(np.unique([i.NAME for i in args])) != len([i.NAME for i in args]):
            print([i.NAME for i in args])
            raise ValueError('Multiple Factor with one name')
        self.initialized_factor_dict = dict(zip([i.NAME for i in args], args))
        

        
    def fillNA(self, dataloader):
        for i, v in self.val.items():
            v[dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_CLOSE').isna()]= np.nan
            self.val[i] = v
        
        

    def fillIndustrial(self, dataloader):
        
        for i, v in self.val.items():
            self.val[i] = utils.fillIndustrialMean(
                dataloader.getIndustry(), v)
    
    def madWinsorize(self):
        
        for i, v in self.val.items():
            self.val[i] = utils.madWinsorize(v)
    
    def weightedStandardize(self, dataloader):
        
        for i, v in self.val.items():
            self.val[i] = utils.weightedStandardize(
                v,  dataloader.getCirculateMarketValue())
    
        
    

    def cal(self, dataloader):
        
        compo_val = []
        if len(self.USED_FACTOR_DICT.keys()) == 0:
            raise ValueError('No used factor')
            
        for i,v in self.USED_FACTOR_DICT.items():
            try:
                if isinstance(v, str):
                    compo_val.append(self.initialized_factor_dict[i].val[v])
                elif isinstance(v, list):
                    for kk in v:
                        compo_val.append(self.initialized_factor_dict[i].val[kk])
                    
            except:
                self.initialized_factor_dict[i].load()
                if isinstance(v, str):
                    compo_val.append(self.initialized_factor_dict[i].val[v])
                elif isinstance(v, list):
                    for kk in v:
                        compo_val.append(self.initialized_factor_dict[i].val[kk])
        
        
        
        for num, i in enumerate(compo_val):
            i.index = pd.to_datetime(i.index)
            if self.IS_STANDARDIZE:
                i = utils.weightedStandardize(i, dataloader.getCirculateMarketValue())
                compo_val[num] = i 
            
            
        val = utils.addFactors(np.array(self.weight), *compo_val)
        self.val = dict(zip(self.name, [val]))



class Size_Comp(SecondaryFactor):
    NAME = 'SIZE_COMP'
    name = ['Size']
    USED_FACTOR_DICT = {'SIZE': 'Size'}
    weight = [1]
    

class Beta_Comp(SecondaryFactor):
    NAME = 'BETA_COMP'
    name = ['Beta']
    USED_FACTOR_DICT = {
        'BETA' : 'Beta'
        }
    weight = [1]
    

class Momentum_Comp(SecondaryFactor):
    NAME = 'MOMENTUM_COMP'
    name = ['Momentum']
    USED_FACTOR_DICT = {
        'RSTR':'RSTR'
        }
    weight = [1] 

class ResidualVolaility_Comp(SecondaryFactor):
    NAME= 'RESIDUALVOL_COMP'
    name = ['ResidualVolatility']
    USED_FACTOR_DICT = {
        'DASTD':'DaStd',
        'CMRA':'CMRA',
        'BETA':'Hsigma'
        }
    weight = [0.74, 0.16, 0.1]
    IS_STANDARDIZE = True


class NonLinearSize_Comp(SecondaryFactor):
    NAME = 'NONLINEARSIZE_COMP'
    name = ['NonLinearSize']
    USED_FACTOR_DICT = {
        'NONLINEARSIZE' :'NonLinearSize'
        
        }
    weight = [1]
   


class BooktoPrice_Comp(SecondaryFactor):
    NAME = 'BOOKTOPRICE_COMP'
    name = ['BooktoPrice']
    USED_FACTOR_DICT = {
        'BOOKTOPRICE': 'BP'
        }
    weight = [1]
    FUNDAMENTAL = True


class Liquidity_Comp(SecondaryFactor):
    NAME = 'LIQUIDITY_COMP'
    name = ['Liquidity']
    USED_FACTOR_DICT ={
        'STO':['STOM', 'STOQ', 'STOA']
        }
    weight = [0.35, 0.35, 0.3]


class EarningsYield_Comp(SecondaryFactor):
    NAME = 'EARNINGSYIELD_COMP'
    name = ['EarningsYield']
    USED_FACTOR_DICT = {
        'ETOP':['CETOP', 'ETOP']
        
        }
    weight = [0.21, 0.11]
    IS_STANDARDIZE = True
    FUNDAMENTAL = True

    
class Growth_Comp(SecondaryFactor):
    NAME = 'GROWTH_COMP'
    name = ['Growth']
    USED_FACTOR_DICT = {
        'EGRF': ['EGRLF', 'EGRSF'],
        'GRO': ['EGRO', 'SGRO']
        }
    weight = [0.18, 0.11, 0.24, 0.47]
    IS_STANDARDIZE = True
    FUNDAMENTAL = True



class Leverage_Comp(SecondaryFactor):
    NAME = 'LEVERAGE_COMP'
    name = ['Leverage']
    USED_FACTOR_DICT = {
        'LEV':['MLEV', 'BLEV'],
        'DTOA': 'DTOA'
        }
    weight = [0.38, 0.27, 0.35]
    IS_STANDARDIZE = True
    FUNDAMENTAL = True



    
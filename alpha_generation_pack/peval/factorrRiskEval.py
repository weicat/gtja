# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 13:38:43 2022

@author: Administrator
"""
from itertools import chain
from functools import reduce
import pandas as pd 

class Evaluation(object):
    
    @staticmethod 
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



class FactorEvaluation(Evaluation):
    
    def __init__(self, barraloader):
        self.barraloader = barraloader
    
    
    def align(self, df1, df2):
        df1 = df1.copy()
        df2 = df2.copy()
        st1 = df1.columns[0]
        if len(st1.split('.')) == 1:
            df1.columns = [(i[2:] + '.' + i[:2]).replace('NE', 'BJ') 
                           for i in df1.columns]
        
        st2 = df2.columns[0]
        if len(st2.split('.')) == 1:
            df2.columns = [(i[2:] + '.' + i[:2]).replace('NE', 'BJ') 
                           for i in df2.columns]
        
            
            
        df1.index = pd.to_datetime(df1.index)
        df2.index = pd.to_datetime(df2.index)
        df1, df2 = Evaluation._align(df1, df2)
        return df1, df2
    
    def getCorrelation(self, df1, df2):
        df1, df2 = self.align(df1, df2)    
        corr = df1.corrwith(df2, axis = 1)
        return corr.mean()
    
    def getRankCorrelation(self, df1, df2):
        df1, df2 = self.align(df1, df2)    
        corr = df1.corrwith(df2, axis = 1, method = 'spearman')
        return corr.mean()
    
    def loopUtils(self, func, df, df_lst, index=None, name = ''):
        res = [func(df, i) for i in df_lst]
        if index is None:
            index = range(len(df_lst))
        return pd.Series(res,index = index, name = name)

    def getRiskReport(self, factor):
        
        lst = []
        
        TESTRISKS = ['Beta', 
                    'BooktoPrice',
                    'EarningsYield',
                    'Growth',
                    'Leverage', 
                    'Liquidity',
                    'Momentum',
                    'NonLinearSize',
                    'ResidualVolatility',
                    'Size']
        
        df_lst = [self.barraloader.getFinalFactor(i) for i in TESTRISKS]    
        ic = self.loopUtils(self.getCorrelation, factor, df_lst, 
                            index = TESTRISKS, name = 'Correlation')
        
        lst.append(ic)
        return pd.concat(lst)
        
        
        
    
    
    
    
    
    
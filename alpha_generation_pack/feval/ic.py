# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:50:50 2022

@author: lth
"""

import numpy as np 
import utils.pandas_utils
from tqdm import tqdm

def setNanFunc(factor, problem_df):
    problem_df = problem_df.fillna(1).astype(int).replace([1, 0], [np.nan, 1])
    problem_df = problem_df[factor.index[0] : factor.index[-1]]
    problem_df, factor = utils.pandas_utils._align(problem_df, factor)
    return factor * problem_df
    
    




def IC(factor, 
       dataloader,
       universe = 'NOBJ',
       method = 'pearson',
       periods = [1],
       remove_st = True,
       remove_suspend = True,
       remove_zt = True,
       remove_dt = True,
       show_progress = True,
       industry_netural = False):

    if universe == 'NOBJ':
        choosed = []
        for i in factor.columns:
            if i.split('.')[1] != 'BJ':
                choosed.append(i)
        
        factor = factor.loc[:, choosed].sort_index(axis = 1)

    
    if remove_st:
        factor = setNanFunc(factor, dataloader.getST())
    if remove_suspend:
        factor = setNanFunc(factor, dataloader.getSuspend())        
    if remove_zt:
        factor = setNanFunc(factor, dataloader.getSPZT())        
    if remove_dt:
        factor = setNanFunc(factor, dataloader.getSPDT())
    
    
    adjclose = dataloader.getAdjClose()
    
    if not industry_netural:
        lst = []
        for i in tqdm(periods, 
                      disable = not show_progress,
                      desc = 'Correlation测试'):
            ret = adjclose.pct_change(peridos = i)
            ret, mytempfac = utils.pandas_utils._align(ret, factor)
            res = ret.corrwith(mytempfac, axis = 1, method = method)
            res.name = i
            lst.append(res)
        
        return lst
    
    if industry_netural:
        res_dic = {}
        industry = dataloader.getIndustry().unstack()
        ind_cls = industry.value_counts().index
        
        for ind in tqdm(ind_cls, 
                        desc = '行业Correlation测试',
                        total = len(ind_cls),
                        disable = not show_progress):
            mytempind = industry[industry == ind].unstack().T
            mytempind, mytempfac = utils.pandas_utils._align(mytempind, factor)
            mytempind = mytempind.isna().astype(int).replace([1,0], [np.nan,1])
            mytempfac = mytempind * mytempfac
            
            lst = []
            for i in tqdm(periods, 
                          disable = True,
                          desc = 'Correlation测试'):
                ret = adjclose.pct_change(periods = i)
                ret, mytempfac2 = utils.pandas_utils._align(ret, mytempfac)
                res = ret.corrwith(mytempfac2, axis = 1, method = method)
                res.name = i
                lst.append(res)
            
            res_dic[ind] = lst
        
        return res_dic
    
    

    
    
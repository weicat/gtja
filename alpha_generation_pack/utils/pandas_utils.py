# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:25:47 2022

@author: lth
"""
from itertools import chain
from functools import reduce
import pandas as pd
import numpy as np 

'''
解决一下inner sum的问题

'''

def getOneHotArr(df, 
                 sortstr_func = lambda x: int(x[2:]),
                 dummy_na = False,
                 sliceArr = True):
    z = df.unstack().value_counts().index
    z = sorted(z, key = sortstr_func)
    z = list(z)
    z.append(np.nan)
    dic = dict(zip(z, range(len(z))))
    transformed = df.apply(lambda x: x.map(dic))
    
    arr = transformed.values
    ncols = arr.max()+1
    out = np.zeros( (arr.size,ncols), dtype=np.uint8)
    out[np.arange(arr.size),arr.ravel()] = 1
    out.shape = arr.shape + (ncols,)
    
    if not sliceArr:
        if dummy_na: 
            return out 
        else:
            return out[:,:,:-1]
    
    else:
        if not dummy_na:
            out = out[:, :, :-1]
        
        tf = out.sum(axis = 1) == 0
        index = np.append([0], (tf[1:] != tf[:-1]).sum(axis = 1))
        sliced_index = np.where(index !=0)[0]
        sliced_index = np.append([0], sliced_index)
        sliced_index = np.append(sliced_index, len(index))
        
        res_lst = []
        choose_lst = []
        
        for num in range(1, len(sliced_index)):

            temp_res = out[sliced_index[num -1]: sliced_index[num], :,:]
            
            temp_choose = np.where(~tf[sliced_index[num]-1 ,:])[0]
            
            choose_lst.append(temp_choose)
            res_lst.append(temp_res[:,:,temp_choose])
        
        return res_lst, [[sliced_index[i-1], sliced_index[i]] for i in range(1, len(sliced_index))], choose_lst
            
        

    



def _align(df1, df2, *dfs, axis = -1):
    dfs_all = [df for df in chain([df1, df2], dfs)]
    if any(len(df.shape) == 1 or 1 in df.shape for df in dfs_all):
        dims = 1
    else:
        dims = 2
    mut_date_range = sorted(reduce(lambda x,y: x.intersection(y), (df.index for df in dfs_all)))
    mut_codes = sorted(reduce(lambda x,y: x.intersection(y), (df.columns for df in dfs_all)))
    if dims == 2:
        if axis == -1:
            dfs_all = [df.loc[mut_date_range, mut_codes] for df in dfs_all]
        elif axis == 1:
            dfs_all = [df.loc[:, mut_codes] for df in dfs_all]
        elif axis == 0:
            dfs_all = [df.loc[mut_date_range, :] for df in dfs_all]

    elif dims == 1:
        dfs_all = [df.loc[mut_date_range, :] for df in dfs_all]
    return dfs_all


def _alignWithNone(*dfs, **kargs):
    
    tf = [not i is None for i in dfs]
    if sum(tf) == 0:
        return [None for i in dfs]
    elif sum(tf) == 1:
        res = [None for i in dfs]
        res[tf.index(True)] = dfs[tf.index(True)]
    
    else:
        z = np.where(tf)[0]
        notNoneDFs = [dfs[i] for i in z]
        tempres = _align(*notNoneDFs, **kargs)
        res = [None for i in dfs]
        for df, i in zip(tempres, z):
            res[i] = df
        
    
    return res


def toValuesWithNone(*dfs):
    return [i if i is None else i.values for i in dfs]


def _sliceIndex(start, end,  *dfs):
    return [i if i is None else i[start: end] for i in dfs]






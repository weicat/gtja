# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:25:47 2022

@author: lth
"""
from itertools import chain
from functools import reduce
import pandas as pd
import numpy as np 


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


def _alignWithNone(*dfs):
    
    tf = [not i is None for i in dfs]
    if sum(tf) == 0:
        return [None for i in dfs]
    elif sum(tf) == 1:
        res = [None for i in dfs]
        res[tf.index(True)] = dfs[tf.index(True)]
    
    else:
        z = np.where(tf)[0]
        notNoneDFs = [dfs[i] for i in z]
        tempres = _align(*notNoneDFs)
        res = [None for i in dfs]
        for df, i in zip(tempres, z):
            res[i] = df
        
    
    return res


def toValuesWithNone(*dfs):
    return [i if i is None else i.values for i in dfs]


def _sliceIndex(start, end,  *dfs):
    return [i if i is None else i[start: end] for i in dfs]


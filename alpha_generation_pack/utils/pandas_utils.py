# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:25:47 2022

@author: lth
"""
from itertools import chain
from functools import reduce
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
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:46:18 2022

@author: lth
"""

from numba import njit
import pandas as pd 
import numpy as np
from utils.pandas_utils import _align


def factorSortAndReduce(f1, f2, reduced_func, sort_axis = -1):
    '''
    f1, f2 是 ndarray
    f2 sort, 把顺序传给f1, 再用reduced function
    sort_axis 是 f2的axis,
    reduced_func 需要numba 可编译
    '''
    if f1.shape != f2.shape:
        raise ValueError('Check factors shape')
    

    @njit
    def looping(arr, idx, func):
        res = np.zeros(shape = (arr.shape[0], arr.shape[1]))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                res[i,j] = func(arr[i,j,:][idx[i,j,:]])
        
        return res
    
    
    
    argsort_arr = np.argsort(f2, axis = sort_axis, kind = 'stable')
    res = looping(f1, argsort_arr, reduced_func)
    return res
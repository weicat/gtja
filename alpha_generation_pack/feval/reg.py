# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:43:20 2022

@author: lth
"""

import utils.pandas_utils
import numpy as np 
import pandas as pd
from tqdm import tqdm

def chunkRegressionResid(arrx, arry, add_constant = True, show_progress = True):
    if add_constant:
        one = np.ones(shape = (arrx.shape[0], arrx.shape[1], 1))
        arrx = np.concatenate((one, arrx), axis = 2)
    arrx = np.ma.masked_array(arrx, mask = np.isnan(arrx))
    arry = np.ma.masked_array(arry, mask = np.isnan(arry))
    if len(arry.shape) == 2:
        arry = np.expand_dims(arry, axis = 2)
    if len(arrx) != len(arry):
        raise ValueError('Check x,y dimension')
    arrxT = arrx.transpose((0,2,1))
    res_arr = np.zeros(shape = (arrx.shape[0], arrx.shape[1]))
    for i in tqdm(range(len(arrx)), disable = not show_progress, desc = '正交中'):
        beta = np.linalg.inv(arrxT[i,:,:].dot(arrx[i,:,:])).dot(arrxT[i,:,:]).dot(arry[i,:,:])
        res = arry[i,:,:] - arrx[i,:,:].dot(beta)
        res_arr[i,:] = res.reshape(-1)
    
    return res_arr
        
    
    
    


def fastOrthogonal(
        factor,
        *factors,
        use_ind = None,
        add_constant = False,
        show_progress = True
        ):
    '''
    暂不支持除了行业以外其他的categorical variable
    '''
    if not use_ind is None:
        z = utils.pandas_utils._align(factor, use_ind, *factors)
        factor = z[0]
        use_ind = z[1]
        factors = z[2:]        
        arr_sl, index_sl, _ = utils.pandas_utils.getOneHotArr(use_ind)
        res_lst = []
        if len(factors) == 0:
            for arr, sted_index in tqdm(zip(arr_sl, index_sl),
                                        disable = not show_progress,
                                        total = len(arr_sl),
                                        desc = '正交中'):
                arrx = arr
                arry = factor.values[sted_index[0]:sted_index[1],:]
                res_lst.append(chunkRegressionResid(arrx, arry, add_constant, False))
        else:
            factors = np.concatenate([np.expand_dims(i.values, axis = 2) 
                                      for i in factors], axis = 2)
            for arr, sted_index in tqdm(zip(arr_sl, index_sl),
                                        disable = not show_progress,
                                        total = len(arr_sl),
                                        desc = '正交中'):
                arrx = np.concatenate([arr, factors[sted_index[0] : sted_index[1]]], axis = 2)
                arry = factor.values[sted_index[0]:sted_index[1],:]
                res_lst.append(chunkRegressionResid(arrx, arry, add_constant, False))
    else:
        z = utils.pandas_utils._align(factor, *factors)
        factor = z[0]
        factors = z[1:]
        if len(factors) == 0:
            return factor
        else:
            arrx = np.concatenate([np.expand_dims(i.values, axis = 2) 
                                      for i in factors], axis = 2)
            arry = factor.values
            res_lst = [chunkRegressionResid(arrx, arry, add_constant, show_progress = show_progress)]
            
            
    res_lst = np.concatenate(res_lst)
    res_lst = pd.DataFrame(res_lst, index = factor.index, columns = factor.columns)
    return res_lst
            
        
        
    
    
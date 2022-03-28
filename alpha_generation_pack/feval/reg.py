# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:43:20 2022

@author: lth
"""

import utils.pandas_utils
import numpy as np 
import pandas as pd
from tqdm import tqdm

def chunkRegressionResid(arrx, arry,
                         weight = None,
                         return_statistics = True,
                         add_constant = True, 
                         show_progress = True):
    if add_constant:
        one = np.ones(shape = (arrx.shape[0], arrx.shape[1], 1))
        arrx = np.concatenate((one, arrx), axis = 2)
    arrx = np.ma.masked_array(arrx, mask = np.isnan(arrx))
    arry = np.ma.masked_array(arry, mask = np.isnan(arry))
    
    
    if len(arry.shape) == 2:
        arry = np.expand_dims(arry, axis = 2)
    if len(arrx) != len(arry):
        raise ValueError('Check x,y dimension')
    
    if not weight is None:
        w = np.ma.masked_array(weight.values, mask = np.isnan(weight.values))
        if len(arrx) != len(w):
            raise ValueError('check weight dimension')
    
    arrxT = arrx.transpose((0,2,1))
    res_arr = np.zeros(shape = (arrx.shape[0], arrx.shape[1]))
    inv_trace_arr = np.zeros(shape = (arrx.shape[0], arrx.shape[-1]))
    beta_arr = np.zeros(shape = (arrx.shape[0], arrx.shape[-1]))
    ddof_arr = np.zeros(shape = (arrx.shape[0],1) )
    sigma_arr = np.zeros(shape = (arrx.shape[0], 1))    
    for i in tqdm(range(len(arrx)), disable = not show_progress, desc = '正交中'):
        if not weight is None:
            inv_arr = np.linalg.inv(arrxT[i,:,:].dot(w).dot(arrx[i,:,:]))
            beta = inv_arr.dot(arrxT[i,:,:]).dot(w).dot(arry[i,:,:])
        else:
            inv_arr = np.linalg.inv(arrxT[i,:,:].dot(arrx[i,:,:]))
            beta = inv_arr.dot(arrxT[i,:,:]).dot(arry[i,:,:])

        res = arry[i,:,:] - arrx[i,:,:].dot(beta)
        res_arr[i,:] = res.reshape(-1)
        
        if return_statistics:
            inv_trace = np.diag(inv_arr)
            ddof = (~res.mask).sum() - arrx.shape[-1]
            sigma = np.nanstd(res)
            inv_trace_arr[i,:] = inv_trace
            beta_arr[i,:] = beta.reshape(-1)
            ddof_arr[i,:] = ddof
            sigma_arr[i,:] = sigma
    if not return_statistics:        
        return res_arr
    else:
        return res_arr, (beta_arr, inv_trace_arr, ddof_arr, sigma_arr)
        
    
    
    


def fastOrthogonal(
        factor,
        *factors,
        weight = None,
        use_ind = None,
        return_statistics = False,
        add_constant = False,
        show_progress = True
        ):
    '''
    暂不支持除了行业以外其他的categorical variable
    '''
    
    fac_names = []
    if len(factors) == 0:
        pass
    else:
        for num, i in enumerate(factors):
            if isinstance(i, pd.DataFrame):
                try:
                    if i.name is None:
                        fac_names.append('factor'+str(num))
                    else:
                        fac_names.append(i.name)
                except:
                    fac_names.append('factor'+str(num))
    
    if not use_ind is None:
        if not weight is None:
            z = utils.pandas_utils._align(factor, use_ind, weight, *factors)
            factor = z[0]
            use_ind = z[1]
            weight = z[2]
            factors = z[3:] 
        else:
            z = utils.pandas_utils._align(factor, use_ind, *factors)
            factor = z[0]
            use_ind = z[1]
            factors = z[2:]        
        arr_sl, index_sl, c_sl, name = utils.pandas_utils.getOneHotArr(use_ind)
        
        fac_names = list(name) + fac_names
        res_lst = []
        
        
        trace_lst = np.zeros(shape = (factor.shape[0], len(fac_names)) )
        trace_lst[:] = np.nan
        beta_lst = np.zeros(shape = (factor.shape[0], len(fac_names)) )
        beta_lst[:] = np.nan
    
        ddof_lst = np.zeros(shape = (factor.shape[0], 1))
        ddof_lst[:] = np.nan
        sigma_lst = np.zeros(shape = (factor.shape[0], 1))
        sigma_lst[:] = np.nan
        
        if len(factors) != 0:
            factors = np.concatenate([np.expand_dims(i.values, axis = 2) 
                                      for i in factors], axis = 2)
        for arr, sted_index, choose in tqdm(zip(arr_sl, index_sl, c_sl),
                                    disable = not show_progress,
                                    total = len(arr_sl),
                                    desc = '正交中'):
            if len(factors) == 0:
                arrx = arr
            else:
                arrx = np.concatenate([arr, factors[sted_index[0] : sted_index[1]]], axis = 2)

            arry = factor.values[sted_index[0]:sted_index[1],:]
            w = None if weight is None else weight.values[sted_index[0]:sted_index[1],:]

            if not return_statistics:
                res_lst.append(chunkRegressionResid(arrx, arry, 
                                                    weight = w,
                                                    return_statistics = return_statistics,
                                                    add_constant = add_constant, 
                                                    show_progress = False))
            else:
                temp_res, stats = chunkRegressionResid(arrx, arry, 
                                                    weight = w,
                                                    return_statistics = return_statistics,
                                                    add_constant = add_constant, 
                                                    show_progress = False)
                res_lst.append(temp_res)
                beta, trace, ddof, sigma = stats
                
                actual_choose = np.append(choose,
                                          np.array(list(range(len(name), len(fac_names) ))))
                
                beta_lst[sted_index[0]:sted_index[1], actual_choose] = beta
                trace_lst[sted_index[0]:sted_index[1], actual_choose] = trace
                ddof_lst[sted_index[0]:sted_index[1], :] = ddof
                sigma_lst[sted_index[0]:sted_index[1], :] = sigma
        
        

    else:
        if not weight is None:
            z = utils.pandas_utils._align(factor, weight, *factors)
            factor = z[0]
            weight = z[1]
            factors = z[2:]
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
            if not return_statistics:
                res_lst = [chunkRegressionResid(arrx, arry, 
                                                    weight = weight.values,
                                                    return_statistics = return_statistics,
                                                    add_constant = add_constant, 
                                                    show_progress = False)]
            else:
                
                temp_res, stats = chunkRegressionResid(arrx, arry, 
                                                    weight = weight.values,
                                                    return_statistics = return_statistics,
                                                    add_constant = add_constant, 
                                                    show_progress = False)
                
                res_lst = [temp_res]
                beta_lst, trace_lst, ddof_lst, sigma_lst = stats
                
                
                
                
            
            
    res_lst = np.concatenate(res_lst)
    res_lst = pd.DataFrame(res_lst, index = factor.index, columns = factor.columns)
    trace_lst = pd.DataFrame(trace_lst, index = factor.index, columns = fac_names)
    beta_lst = pd.DataFrame(trace_lst, index = factor.index, columns = fac_names)
    ddof_lst = pd.DataFrame(ddof_lst, index = factor.index, columns = ['degrees of freedom'])
    sigma_lst = pd.DataFrame(sigma_lst, index = factor.index, columns = ['residual_std'])
    
    if return_statistics:
        return res_lst, (beta_lst, trace_lst, ddof_lst, sigma_lst)
    else:
        return res_lst
    
        
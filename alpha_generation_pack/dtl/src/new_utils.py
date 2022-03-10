import numpy as np 
from numba import njit, guvectorize, float64, int64, boolean
from itertools import chain
from functools import reduce
import pandas as pd 
import statsmodels.api as sm



def orthogonalOnce(Y, *X):
    lst = [i for i in X]
    lst = [Y] + lst
    res = pd.concat(lst,axis = 1).dropna()
    if len(res) ==0:
        return pd.Series(np.nan, index = Y.index, name = Y.name)
    Ynew = res.iloc[:,0]
    Xnew = res.iloc[:,1:]
    
    result = sm.OLS(Ynew, sm.add_constant(Xnew)).fit()
    orthogonaled = result.resid
    orthogonaled = pd.Series(orthogonaled, index = Ynew.index, name = Ynew.name)
    df = pd.concat([orthogonaled, Y], axis = 1)
    return df.iloc[:,0]

def weightedStandardize(df, weight):
     mean = (((weight[~df.isna()].T/weight[~df.isna()].T.sum()).T)*df).sum(axis = 1)
     res = (df.T - mean)/df.T.std()
     return res.T
 

def madWinsorize(df, winsorize = 5):
    

    
    med = df.T.median()
    mad = np.abs(df.T - df.T.median(skipna = 'drop')).median(skipna = 'drop')
    p = df.T.clip(lower = med - 5 * mad, upper = med + 5 * mad, axis = 1)
    return p.T
def orthogonal(Y, *X):
    lst = []
    res = _align(Y, *X)
    Y = res[0]
    X = res[1:]
    for i in range(len(Y)):
        y = Y.iloc[i,:]
        x = [j.iloc[i,:] for j in X]
        lst.append(orthogonalOnce(y, *x))
    
    return pd.concat(lst,axis = 1).T


#

def getIndustrialMean(industry, factor):
    industry = industry.replace('', np.nan)
    industry, factor = _align(industry, factor)
    lst = []
    for i in range(len(factor)):
        f = factor.iloc[i,:]
        i = industry.iloc[i,:]
        t = pd.concat([f, i], axis =1).dropna()
        t.columns = ['factor', 'ind']
        c = t.groupby('ind').mean()    
        lst.append(c)
    
    lst = pd.concat(lst, axis = 1)
    lst.columns = industry.index
    return lst.T


def fillIndustrialMean(industry, factor):
    industry = industry.replace('', np.nan)
    industry, factor = _align(industry, factor)
    lst = []
    for i in range(len(factor)):
        f = factor.iloc[i,:]
        ind = industry.iloc[i,:]
        t = pd.concat([f, ind], axis =1).dropna()
        t.columns = ['factor', 'ind']                
        c = t.groupby('ind').mean()
        if len(c) == 0:
            lst.append(f)
            continue
        
        
        filled = ind[f.isna()].dropna()
        
        added = set(filled.values) - set(c.index)
    
        if len(added) !=0 :
            for add in added:
                c.loc[add, 'factor'] = np.nan
        
        
        fc = f.copy()
        fc[filled.index] = c.loc[filled.values].values.reshape(-1)

        lst.append(fc)
    
    lst = pd.concat(lst, axis = 1)
    return lst.T    

    
@njit
def addFactorsNumba(w, args):
    row, col = args[0].shape
    
    arr = np.zeros(shape = (row , col))
    arglen = len(args)
    for i in range(row):
        for j in range(col):
            e = np.array([args[jj][i,j] for jj in range(arglen)])
            tfarr = np.array([np.isnan(element) for element in e])
            copyed = []
            for k in range(len(tfarr)):
                if tfarr[k] == False:
                    copyed.append(1)
                else:
                    copyed.append(0)
            tfarr = np.array(copyed)
            tf = np.where(tfarr == 1)
            if len(tf[0]) == 0:
                t = np.nan
            else:
                t = np.sum(e[tf] * w[tf])
                t = t/np.sum(w[tf])
            
            arr[i,j] = t

    return arr

def addFactors(w, *args):
    if len(w) == 1:
        return args[0]
    r = _align(*args)
    inputargs = [np.ascontiguousarray( i.values ) for i in r]
    res = addFactorsNumba(w, inputargs)
    res = pd.DataFrame(res, index = r[0].index, columns = r[0].columns)
    return res


def getWeight(alpha, windowSize):
    
    wghts = (1-alpha)**np.arange(windowSize)
    wghts /= wghts.sum()
    return wghts[::-1]


@njit
def CAPMRegress(arr, market_arr, weight):
    '''
    这里还是要在内部进行copy后再计算，
    因为 sliding window arr 和 broadcast arr
    都是unwriteable， 强行改变write会导致unpredictable behavior(例如braodcast
                                                      修改一个元素会造成整个
                                                      ndarray的此列元素变掉)
    
    还有一个问题就是numba不支持 mask_array，
    此算法目前来看是唯一解决办法
    
    std和之前算的略有区别
    
    '''

    arr = arr.copy()
    weight = weight.copy()
    
    res = np.empty(shape = arr.shape[0])
    res[:] = np.nan
    
    res2 = np.empty(shape = arr.shape[0])
    res2[:] = np.nan    
    
    res3 = np.empty(shape = arr.shape[0])
    res3[:] = np.nan    
    
    nan_index = np.isnan(arr)

    
    for i in range(weight.shape[1]):
        weight[:,i][nan_index[:, i]] = 0.
        arr[:,i][nan_index[:, i]] = 0
        
    chosen = np.where(np.sum(weight, axis = 1) >= 0.5 )[0]
    weight = weight[chosen, :]
    arr = arr[chosen, :]
    y00 = (weight * arr).sum(axis = 1)
    y01 = (weight * market_arr * arr).sum(axis = 1)
    
    x00 = (market_arr * market_arr * weight).sum(axis = 1 )
    x01 = -(market_arr * weight).sum(axis = 1)
    x10 = x01
    x11 = weight.sum(axis = 1)
    
    div = x00 * x11 - x10*x01
    alpha = (x00 * y00 + x01 * y01)/div
    beta =  (x10 * y00 + x11 * y01)/div
    
    res[chosen] = beta
    res2[chosen] = alpha
    resid = arr - (alpha.reshape(-1, 1) + beta.reshape(-1, 1) * market_arr)
    res3[chosen] = np.array([np.std(resid[i,:]) for i in range(resid.shape[0]) ])
    
    return res, res2, res3


@njit
def numbaCMRA(arr, weight, month_days, month_len):
    res = np.empty(shape = arr.shape[0])
    res[:] = np.nan

    nan_index = np.isnan(arr)
    weight = weight.copy()
    for i in range(weight.shape[1]):
        weight[:,i][nan_index[:, i]] = 0.    
    
    chosen = np.where(np.sum(weight, axis = 1) >= 0.5 )[0]
    weight = weight[chosen, :]
    sliced_arr = arr[chosen, :]
    
    
    res_arr = np.ones(shape = sliced_arr.shape[0])
    for s in range(sliced_arr.shape[0]):
        temp_arr = np.ones(shape = month_len)
        for m in range(1, month_len + 1):
            temp_arr[m - 1] = np.nansum(sliced_arr[s, -month_days * m :])
        res_arr[s] = np.max(temp_arr) - np.min(temp_arr)
        
    
    res[chosen] = res_arr
    return res
    
    
    # return np.max(res) - np.min(res)
    


@guvectorize([( float64[:, :, :], 
                float64[:, :, :], 
                float64[:,:], 
                float64[:,:],
                float64[:,:],
                float64[:,:])], 
              '(n, m, p),(n, q, p),(m, p) -> (n, m), (n, m), (n, m)')
def CAPMRegressVectorized(arrs, market_arrs, weight, resbeta, resalpha, resresid):
    for i in range(arrs.shape[0]):
        res, res2, res3 = CAPMRegress(arrs[i,:,:], market_arrs[i,:, :], weight)
        resbeta[i]  = res
        resalpha[i] = res2
        resresid[i] = res3


@guvectorize([( float64[:, :, :], 
                float64[:, :], 
                int64, 
                int64,
                float64[:, :])], 
              '(n, m, p),(m, p),(),() -> (n, m)')
def CMRAVectorized(arrs, weight, month_days, month_len, res):
    for i in range(arrs.shape[0]):
        res[i] = numbaCMRA(arrs[i,:,:], weight, month_days, month_len)
    



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

def getExponentialWeight(window, half_life):
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] / np.sum(exp_wt)




@njit
def weightedSum(arr, weight):
    res = np.empty(shape = arr.shape[0])
    res[:] = np.nan
    
    nan_index = np.isnan(arr)
    weight = weight.copy()
    arr = arr.copy()
    for i in range(weight.shape[1]):
        weight[:,i][nan_index[:, i]] = 0.
        arr[:, i][nan_index[:, i]] = 0
    chosen = np.where(np.sum(weight, axis = 1) >= 0.5 )[0]
    temp = (weight[chosen, :] * arr[chosen, :]).sum(axis = 1)
    res[chosen] = temp
    
    return res

@njit
def weightedVar(arr, weight):
    res = np.empty(shape = arr.shape[0])
    res[:] = np.nan
    
    nan_index = np.isnan(arr)
    weight = weight.copy()
    arr = arr.copy()
    for i in range(weight.shape[1]):
        weight[:,i][nan_index[:, i]] = 0.
        arr[:, i][nan_index[:, i]] = 0
    chosen = np.where(np.sum(weight, axis = 1) >= 0.5 )[0]
    
    mean = np.array([np.nanmean(arr[i,:]) for i in chosen ])
    mean = mean.reshape(-1, 1)
    temp = (weight[chosen, :] * (arr[chosen, :] - mean)**2 ).sum(axis = 1)
    res[chosen] = temp
    
    return res


@guvectorize([( float64[:, :, :], 
                float64[:,:], 
                float64[:,:])], 
              '(n, m, p),(m, p) -> (n, m)')
def weightedSumVectorized(arrs, weight, res):
    
    for i in range(arrs.shape[0]):
        res[i] = weightedSum(arrs[i,:,:], weight)
         
@guvectorize([( float64[:, :, :], 
                float64[:,:], 
                float64[:,:])], 
              '(n, m, p),(m, p) -> (n, m)')
def weightedVarianceVectorized(arrs, weight, res):
    
    for i in range(arrs.shape[0]):
        res[i] = weightedVar(arrs[i,:,:], weight)

def weightedStdVectorized(arrs, weight):
    return np.sqrt(weightedVarianceVectorized(arrs, weight))



def sumVectorized(arrs):
    weight = np.ones(shape = [arrs.shape[1], arrs.shape[2]])
    return weightedSumVectorized(arrs, weight)

@njit
def slopeRegress(arr, intercept = True):
    start = 1
    
    y = arr[np.where(~np.isnan(arr))]
    if len(y) < len(arr) * 0.5:
        return np.nan
    
    x = np.arange(start, start + len(y))
        
    if not intercept:
        return np.sum(x * y)/np.sum( x * x)
    
    else:
        n = len(y)
        x1 = np.sum(x)
        x2 = np.sum(x*x) 
        x3 = x1**2
        
        divisor = n * x2- x3
        xy = np.sum(x*y)
        y1 = np.sum(y)
        temp = n * xy - x1 * y1

        return temp/divisor        
    

@guvectorize([(float64[:,:], boolean, float64[:])], '(n, m), () -> (n)')
def slopeVectorized(x, intercept, res):
    for i in range(x.shape[0]):
        res[i] = slopeRegress(x[i,:], intercept)
    
            



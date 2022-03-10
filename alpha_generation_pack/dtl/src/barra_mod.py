'''
有一些停牌股可能算的不对
这个要在加在available stock universe里面
'''
import numba as nb
import pandas as pd
import numpy as np 
# from utils import regressCAPM, align, alignNa, rolling_apply, weightedMean, weightedStd, standardWinsorize, standardize
# from utils import getExponentialWeight
# import utils
import joblib
import statsmodels.api as sm
from tqdm import tqdm
import dtl.src.new_utils as utils



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def beta(       r, rm,
                half_life = 63,
                window = 252
                ) : 
    r, rm = utils._align(r, pd.DataFrame(rm))
    weights = utils.getExponentialWeight(window, half_life)
    
    r_slide = np.lib.stride_tricks.sliding_window_view(
        r.values, window_shape = window, axis = 0
        )
    rm_slide = np.lib.stride_tricks.sliding_window_view(
        rm.values.reshape(-1, 1), window_shape = window, axis = 0
        )
    
    weights = np.vstack([weights] * r.shape[1])
    
    
    beta, _, hsigma = utils.CAPMRegressVectorized(r_slide, rm_slide, weights)
    
    beta = pd.DataFrame(beta, 
                        index = r.index[window - 1 :],
                        columns = r.columns)

    hsigma = pd.DataFrame(hsigma, 
                        index = r.index[window - 1 :],
                        columns = r.columns)

    
    
    return beta, hsigma


#动量fillna 0
def momentum(r,
             longlag = 504,
             half_life = 126,
             riskfree = 0.03/252):

    weight = utils.getExponentialWeight(longlag, half_life)
    obj = np.log(1 + r) - np.log( 1 + riskfree)
    obj_slide = np.lib.stride_tricks.sliding_window_view(
        obj, window_shape = longlag, axis = 0
        )    
    weights = np.vstack([weight] * obj.shape[1])
    c = utils.weightedSumVectorized(obj_slide, 
                                    weights)
    
    return c

    
#去掉na
def dastd(r,
          window = 252,
          half_life = 42,
          riskfree = 0.03/252):
    rc = r - riskfree
    weight = utils.getExponentialWeight(window, half_life)
    weight = np.vstack([weight] * rc.shape[1])
    rc = np.lib.stride_tricks.sliding_window_view(
        rc.values, window_shape = window, axis = 0
        )    
    res = utils.weightedStdVectorized(rc, weight)
    
    return res
    


def cmra(r, 
         month_len = 12,
         month_days = 21,
         riskfree = 0.03/252
         ):
    window = month_len * month_days
    exr = np.log(1 + r) - np.log(1 + riskfree)
    weights = np.array([1]* (window))
    weights = weights/weights.sum()
    weights = np.vstack([weights] * exr.shape[1])


    exr_slide = np.lib.stride_tricks.sliding_window_view(
        exr.values, window_shape = month_len * month_days, axis = 0
        )
    
    cmra = utils.CMRAVectorized(
            exr_slide,
            weights,
            month_days,
            month_len,
            )
    
    cmra = pd.DataFrame(cmra,
                 index = exr.index[window - 1 :],
                 columns = exr.columns)
    
    return cmra
    
    


def nonlinearsize(
        size,
        wls = False,
        ):

    size = np.log(size)
    if wls:
        def nonlinearRegress(df):
            result = sm.WLS(df.dropna() **3 , sm.add_constant(df.dropna()), weight = np.sqrt(np.exp(df).dropna()) ).fit()
            return pd.Series(result.resid, index = df.index)
    else:
        def nonlinearRegress(df):
            result = sm.OLS(df.dropna() **3 , sm.add_constant(df.dropna()) ).fit()
            return pd.Series(result.resid, index = df.index)
    
    nonlinearsize = size.apply(nonlinearRegress, axis = 1)
    
    nonlinearsize = nonlinearsize * (-1)
    
    return nonlinearsize



def sto_func(turnover):
    window = 21
    window_lst = [21, 63, 252]
    scaled_lst = [1, 3, 12]
    lst = []
    for window, scaled in zip(window_lst, scaled_lst):
        slided_turnover = np.lib.stride_tricks.sliding_window_view(
            turnover.values, window_shape = window, axis = 0)
        res = utils.sumVectorized(slided_turnover)
        res = np.log(res/scaled)
        res = pd.DataFrame(res, columns = turnover.columns, 
                            index = turnover.index[window - 1:]).replace([
                                np.inf, -np.inf], np.nan)
        lst.append(res)
    return lst

def gro_func(rolling_obj):
    window = 20
    index, arrs = rolling_obj.getFixedWindowArr(window = window)
    res = utils.slopeVectorized(arrs, False)
    res = res/np.nanmean(arrs, axis = 1)
    output = pd.Series(res, index = index).replace(np.nan, 'P')
    return output


def etop_func(rolling_obj):
    window = 4
    index, arrs = rolling_obj.getFixedWindowArr(window = window)
    res = np.nanmean(arrs, axis = 1)
    output = pd.Series(res, index = index)
    return output

def egrf_func(rolling_obj, window):
    index, arrs = rolling_obj.getFixedWindowArr(window = window)
    t = (arrs[:, -1] - arrs[:, 0]) /np.abs(arrs[:,0])
    return pd.Series(t, index = index)
    



def MLEV_func(MarketValue,StockPriorSharesOtherEquity,StockPriorSharesDebttopay,LongTermDebt):
    ME = MarketValue
    PE = StockPriorSharesOtherEquity + StockPriorSharesDebttopay
    LD = LongTermDebt * 1
    MLEV = (ME+PE+LD)/ME
    return MLEV

def BLEV_func(TotalAsset,TotalLiability,StockPriorSharesOtherEquity,StockPriorSharesDebttopay,LongTermDebt):
    TA = TotalAsset * 1
    TD = TotalLiability * 1
    BE = TA - TD
    PE = StockPriorSharesOtherEquity + StockPriorSharesDebttopay
    LD = LongTermDebt * 1
    BLEV = (BE+PE+LD)/BE
    return BLEV





    
    
    

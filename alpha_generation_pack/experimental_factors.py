import pandas as pd 
from dtl.src.loader import DataLoader
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from utils.pandas_utils import _align
from utils.numba_utils import factorSortAndReduce
from numba import njit




import datetime
z = datetime.datetime.now()

dataloader = DataLoader(r'E:\DataBase')
close = dataloader.read('WINDDB','calculated', 'PVProcess', 'S_DQ_CLOSE')
high = dataloader.read('WINDDB','calculated', 'PVProcess', 'S_DQ_HIGH')
low = dataloader.read('WINDDB','calculated', 'PVProcess', 'S_DQ_LOW')
preclose = dataloader.read('WINDDB','calculated', 'PVProcess', 'S_DQ_PRECLOSE')




chosing_percent = 0.7
rolling_window = 160
ret = close/preclose - 1
ret_range = high / low - 1

ret, ret_range = _align(ret, ret_range)
rolled_ret = sliding_window_view(ret, rolling_window, axis = 0)
rolled_range = sliding_window_view(ret_range, rolling_window, axis = 0)

@njit
def reduced_function(arr):
    return arr[:int(rolling_window * chosing_percent)].sum()

res = factorSortAndReduce(rolled_ret, rolled_range, reduced_function)
res = pd.DataFrame(res, index = ret.index[rolling_window -1 :], columns = ret.columns)
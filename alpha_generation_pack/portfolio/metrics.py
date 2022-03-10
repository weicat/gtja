# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:33:09 2022

@author: Administrator
"""
import numpy as np 



def getMyMetrics(adjusted, cross, turn = 0.2, scale = 252):
    df = adjusted
    daily_ret = df.mean() * scale
    close = (1 + df).cumprod()
    dd = (close/close.cummax() - 1).cummin()[-1]
    sharpe = df.mean()/df.std() * np.sqrt(scale)
    
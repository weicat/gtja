# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:22:12 2022

@author: Administrator
"""



'''
不要加log！！
'''



from dtl.src.loader import DataLoader
from factors.modelbased_factor import AlphaNetFactor_MinuteFactorV2
import os
import pandas as pd 
import datetime
#%%训练
# wholetime = ('2017-05-01', '2022-01-10')
dataloader = DataLoader()
traintime_lst = [
    ('2017-07-10', '2019-07-01'),
    ('2018-01-01', '2020-01-01'),
    ('2018-07-01', '2020-07-01'),
    
    ]
valtime_lst = [
    ('2019-07-10', '2020-07-10'),
    ('2020-01-10', '2021-01-10'),
    ('2020-07-10', '2021-07-10')    
    ]



for traintime, valtime in zip(traintime_lst, valtime_lst):
    f = AlphaNetFactor_MinuteFactorV2(r'H:/alpha_generation_pack/factors/min_model/alphanetmodel', dataloader)
    f.feature_rolling_window = 5
    f.predict_len = 1
    wholetime = str(pd.to_datetime(traintime[0]) - datetime.timedelta(days = 10))[:10], valtime[-1]
    f.train(wholetime, traintime, valtime)


#%%
testtime_lst = [
    ('2020-07-11', '2021-01-10'),
    ('2021-01-11', '2021-07-10'),
    ('2021-07-11', '2022-01-10'),
    ]

# model_name = [
#     '101712',
#     '103250',
#     '104822',
#     '110825'
#     ]

m_path = r'H:/alpha_generation_pack/factors/min_model/alphanetmodel'


pred = []
for j, i in enumerate(os.listdir(m_path)):
    
    f = AlphaNetFactor_MinuteFactorV2(os.path.join(m_path, 
                                    i),
                        dataloader)
    f.feature_rolling_window = 5
    f.shape = (None, 5 * 16, 16)
    
    pred.append(f.predict(testtime_lst[j]))
pred = pd.concat(pred)
pred.to_csv('forecast1_min_nozdt.csv')    

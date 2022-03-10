# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:22:12 2022

@author: Administrator
"""

from dtl.src.loader import DataLoader
from factors.modelbased_factor import AlphaNetFactor, AlphaNetFactorV2
import os
import pandas as pd 
#%%训练
wholetime = ('2015-01-01', '2022-01-10')
dataloader = DataLoader()
traintime_lst = [
    ('2015-01-01', '2019-01-01'),
    ('2015-07-01', '2019-07-01'),
    ('2016-01-01', '2020-01-01'),
    ('2016-07-01', '2020-07-01'),
    
    ]
valtime_lst = [
    ('2019-01-10', '2020-01-10'),
    ('2019-07-10', '2020-07-10'),
    ('2020-01-10', '2021-01-10'),
    ('2020-07-10', '2021-07-10')    
    ]



for traintime, valtime in zip(traintime_lst, valtime_lst):
    f = AlphaNetFactorV2('', dataloader)
    
    f.predict_len = 5
    f.train(wholetime, traintime, valtime)


#%%
testtime_lst = [
    ('2020-01-11', '2020-07-10'),
    ('2020-07-11', '2021-01-10'),
    ('2021-01-11', '2021-07-10'),
    ('2021-07-11', '2022-01-10'),
    ]

model_name = [
    '101712',
    '103250',
    '104822',
    '110825'
    ]


pred = []
for testtime, model in zip(testtime_lst, model_name):
    
    m_path = r'H:/alpha_generation_pack/factors/models/alphanetmodel'
    f = AlphaNetFactorV2(os.path.join(m_path, 
                                    'alphanetV3_{t}.h5'.format(t = model)),
                        dataloader)
    pred.append(f.predict(testtime))
pred = pd.concat(pred)
pred.to_csv('forecast5_nozdt.csv')    

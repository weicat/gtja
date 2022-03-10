# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:18:07 2022

@author: Administrator
"""

import os
import pandas as pd 
from tqdm import tqdm

path1 = r'Z:\LuTianhao\Barra'
path2 = r'Z:\LuTianhao\StaticBarra_CopyedFrom20220302'



def copyFile(foldername, filename):
    df1 = pd.read_pickle(os.path.join(path1, foldername, filename))
    df2 = pd.read_pickle(os.path.join(path2, foldername, filename))
    updated = df1.iloc[-1,:]
    new = df2.append(updated)
    new = new[~new.index.duplicated(keep = 'first')]
    new.to_pickle(os.path.join(path2, foldername, filename))

for folder in tqdm(os.listdir(path1)):
    for file in os.listdir(os.path.join(path1, folder)):
        copyFile(folder, file)
    

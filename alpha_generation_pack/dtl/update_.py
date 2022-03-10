# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:27:36 2022

@author: Administrator
"""

#加路径防止自动化程序报错
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


import dtl.src.connection
import dtl.src.loader
import dtl.src.pipe
import datetime





database_path = r'Z:\LuTianhao\DataBase'
factor_root_path = r'Z:\LuTianhao\Barra'

print('TASK START AT {t}'.format(t = str(datetime.datetime.now())))



conn_dict = {'WINDDB':dtl.src.connection.WindDBData(),
             'TSDB':dtl.src.connection.TinySoftData()}


dataloader = dtl.src.loader.DataLoader(database_path = database_path)


datapipe = dtl.src.pipe.UpdateDataBase(database_path, conn_dict)
barrapipe = dtl.src.pipe.BarraFactorComp(factor_root_path, dataloader)


print('------pipe data --------')
datapipe.run()
print('------pipe barra---------')
barrapipe.run()
print('Finish')


for key in list(globals().keys()):
     if (not key.startswith("__")):
         globals().pop(key) 

del key


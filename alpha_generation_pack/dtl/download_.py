# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:27:36 2022

@author: Administrator
"""

import src
import datetime



database_path = r'Z:\LuTianhao\DataBase'
factor_root_path = r'Z:\LuTianhao\Barra'


print('TASK START AT {t}'.format(t = str(datetime.datetime.now())))


conn_dict = {'WINDDB':src.connection.WindDBData(),
             'TSDB':src.connection.TinySoftData()}


dataloader = src.loader.DataLoader(database_path = database_path)


datapipe = src.pipe.CreateDataBase(database_path, conn_dict)
barrapipe = src.pipe.BarraFactorComp(factor_root_path, dataloader)


# print('------pipe data --------')
# datapipe.run()
# print('------pipe barra---------')
# barrapipe.run()
# print('Finish')
        

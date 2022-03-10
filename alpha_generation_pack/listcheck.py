# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:59:17 2022

@author: Administrator
"""




import pandas as pd 

df = pd.read_excel(
    r'C:\Users\Administrator\Documents\WeChat Files\wxid_ypxbs9g5n95322\FileStorage\File\2022-03\日内交易策略报表-20220309.xlsx',sheet_name = '持仓', index_col = 0)


_in = pd.read_excel(
    r'C:\Users\Administrator\Documents\WeChat Files\wxid_ypxbs9g5n95322\FileStorage\File\2022-03\日内交易策略报表-20220309.xlsx',sheet_name = '7557010借出持仓', index_col = 0)

_out = pd.read_excel(
    r'C:\Users\Administrator\Documents\WeChat Files\wxid_ypxbs9g5n95322\FileStorage\File\2022-03\日内交易策略报表-20220309.xlsx',sheet_name = '借入持仓', index_col = 0)
df = df[df['组合编号'] == 7557010][['交易市场', '证券代码', '持仓']]


_in = _in.iloc[:,[0,1,8]]
_in = _in.set_index('Unnamed: 1')
_in = _in[_in['市值']!=0]
_in = _in.groupby(_in.index).sum()
df = df.set_index('证券代码')



c = pd.concat([df['持仓'], _in['数量']],axis = 1)
chicang = c.fillna(0).sum(axis = 1)
chicang.loc[510500] = chicang.loc[510500] - 3640000
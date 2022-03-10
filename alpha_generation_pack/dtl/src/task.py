# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:13:09 2022

@author: Administrator
"""


import dtl.src.winddb as wdb
import dtl.src.tinysoftdb as tsdb
import pandas as pd 
import os
import numpy as np
import datetime

'''
更新底表用append
衍生表每天算一遍

底表周中append
周末全部更新一遍
'''

'''
现在的Download Process Task 仅支持 task内部，不支持跨task的
Process Task可以跨task使用
'''


'''
可以更加细分 task的dependency(比如只依赖某个task的某个小的Data)
'''



class BaseTask(object):
    NAME = None
    USED_DATA_LST = None
    USED_TASK_LST = None
    DB = None 
    
    
    def __init__(self, conn, *args,
                 root_path = r'Z:\LuTianhao\DataBase'):
        self.conn = conn
        self.initialized_tasks = args
        # self.download_start = download_start
        self.root_path = root_path
        self.initialize()
    
        self.__DONE = False
        self.LOADED_DICT = {}
        self.SAVED_DICT = {}
    
    def cleanRaw(self):
        self.LOADED_DICT = {}
        
    def cleanCal(self):
        self.SAVED_DICT = {}
    
    def close(self):
        self.cleanRaw()
        self.cleanCal()
    
    
    def getStatus(self):
        return self.__DONE
    
    def setDone(self):
        self.__DONE = True

    def check(self):
        
        if self.NAME is None or (self.USED_DATA_LST is None and self.USED_TASK_LST is None) or self.DB is None:
            
            print(self.NAME)
            print(self.USED_TASK_LST)
            print(self.USED_DATA_LST)
            print(self.DB)
            raise ValueError('Check the set task list')
    
    
    def setcls(self):
        
        self.data_cls_dict = {}
        self.task_cls_dict = {}
        if not self.USED_DATA_LST is None:
            for i in self.USED_DATA_LST:
                self.data_cls_dict[i.name] = i(self.conn)
        if not self.USED_TASK_LST is None:
            
            '''
            task 要跟 input 对齐
            '''
            for i, j in zip(self.USED_TASK_LST, self.initialized_tasks) :
                if not isinstance(j, i):
                    raise ValueError('Check input')
                self.task_cls_dict[i.NAME] = j       
        
    def setpath(self):
        
        self.folder_name = self.NAME
        self.download_root_path = os.path.join(self.root_path, self.DB)

    
    def initialize(self):
        
        self.check()
        self.setcls()
        self.setpath()
        
        
    
 
    
    def loadSpec(self, raw, name):
        
        if raw == True:
            self.LOADED_DICT[name] = pd.read_pickle(
                os.path.join(self.getDownLoadPath(),
                             name + '.pickle'
                             )
                )
        else:
            self.SAVED_DICT[name] = pd.read_pickle(
                os.path.join(self.getCalculatedPath(),
                             name + '.pickle'
                             )
                )
            
            
    def loadTask(self, task_name, data_name):
        t = self.task_cls_dict[task_name] 
        if isinstance(t, ProcessTask):
            try:
                return t.SAVED_DICT[data_name]
            except:
                print('Check Pipeline dir {t}, ignore if you have a skip task'.format(
                    t = data_name))
                t.loadSpec(False, data_name)
                return t.SAVED_DICT[data_name]
        
        elif isinstance(t, DownloadTask):
            try:
                return t.LOADED_DICT[data_name]
            except:
                print('Check Pipeline dir {t}, ignore if you have a skip task'.format(
                    t = data_name))
                t.loadSpec(True, data_name)
                return t.LOADED_DICT[data_name]
        else:
            raise ValueError('Unrecognized Task')        
    

class DownloadTask(BaseTask):
    download_start = '20100101'
    
    
    def getDownLoadPath(self):
        t = os.path.join(self.download_root_path, 'raw', 
                                     self.folder_name)
        if not os.path.exists(t):
            os.makedirs(t)
        return t
    
    def download(self):
        
        for i in self.data_cls_dict.keys():
            df = self.data_cls_dict[i].download(self.download_start)
            df.to_pickle(
                os.path.join(self.getDownLoadPath(), i  + '.pickle')
                )
            self.LOADED_DICT[i] = df
        
    def getUpdatedDF(self, df, init_cls):
        chosen_field = init_cls.getUpdateCalendarColumn
        start_field = df[chosen_field].max()
        return init_cls.update(start_field)
    
    def dropDuplicated(self, df, init_cls):
        unique_index = init_cls.unique_index
        temp = df.set_index(unique_index)        
        temp = temp[~temp.index.duplicated(keep = 'last')]
        return temp.reset_index()
    
    def updateOne(self, init_cls):
        
        file_name = init_cls.name
        
        if file_name in self.LOADED_DICT.keys():
            df = self.LOADED_DICT[file_name]
        
        else:
            df = pd.read_pickle(os.path.join(self.getDownLoadPath(),
                                    file_name + '.pickle'))
        
        res =  self.dropDuplicated(df.append(self.getUpdatedDF(df, init_cls)), init_cls)
        
        self.updated_records = len(res) - len(df)
        return res
        

    def update(self):
        for i in self.data_cls_dict.keys():
            df = self.updateOne(self.data_cls_dict[i])
            file_name = self.data_cls_dict[i].name
            df.to_pickle(
                os.path.join(self.getDownLoadPath(), file_name + '.pickle')
                )
            self.LOADED_DICT[file_name] = df

    



class ProcessTask(BaseTask):
    
    
    
    
        
    def unstack(self, df, sep_names, index_names):
        df = df.set_index(index_names)
        df = df[~df.index.duplicated(keep = 'last')]
        df = df[sep_names]
        _dict = {}
        for i in sep_names:
            _dict[i] = df[i].unstack()
        
        return _dict    
    
    def getCalculatedPath(self):
         t =  os.path.join(self.download_root_path, 'calculated', 
                                      self.folder_name)
         if not os.path.exists(t):
             os.makedirs(t)
         return t    
    
    def process(self):
        raise NotImplementedError('Derived Sheet needs to implement process function')
        
    
    def save(self):
        dic = self.process()
        self.SAVED_DICT = dic
        if len(dic) == 0 or dic is None:
            pass
        else:
            for i in dic.keys():
                dic[i].to_pickle(
                    os.path.join(self.getCalculatedPath() , i + '.pickle')
                    )

    
    
    @staticmethod
    def fillSheetByEntryRemove(close, 
                               choosed,
                               code_column = 'S_INFO_WINDCODE',
                               entry_column = 'ENTRY_DT', 
                               remove_column = 'REMOVE_DT',
                               val_column = None):
        
        sheet = close.copy()
        sheet[:] = np.nan
        
        code_pos = choosed[code_column].map(dict(zip(close.columns, 
                                                   range(len(close.columns))
                                                   )))
        choosed = choosed[~code_pos.isna()]
        code_pos = code_pos.dropna().values.astype(int)
        
        entry_pos = np.searchsorted(close.index, 
                                    pd.to_datetime(choosed[entry_column]))
        exit_pos = np.searchsorted(close.index,
                                   pd.to_datetime(choosed[remove_column].replace(
                                       [None], close.index[-1]))
            )
        arr = sheet.values
        if val_column is None:
            concat_lst = [code_pos, entry_pos, exit_pos]
        else:
            concat_lst = [code_pos, entry_pos, exit_pos, choosed[val_column].values]
                    
        t = pd.concat([pd.Series(i) for i in concat_lst], axis = 1)
        tt = t.sort_values(by = [0, 1, 2]).set_index([0, 1])    
        tt = tt[~tt.index.duplicated(keep = 'last')] 
        val = tt.reset_index().values
        
        if val_column is None:
            code_pos, entry_pos, exit_pos = val[:,0], val[:,1], val[:,2]
            for _code, _entry, _exit in zip(code_pos, entry_pos, exit_pos):
                arr[_entry: (_exit + 1), _code] = True
        else:
            code_pos, entry_pos, exit_pos, filled_value = val[:,0], val[:,1], val[:,2], val[:,3]
            try:
                for _code, _entry, _exit, _fill in zip(code_pos, entry_pos, exit_pos, filled_value):
                    arr[_entry: (_exit + 1), _code] = _fill
            except:
                arr = arr.astype(object)
                for _code, _entry, _exit, _fill in zip(code_pos, entry_pos, exit_pos, filled_value):
                    arr[_entry: (_exit + 1), _code] = _fill
        
        
        
        sheet = pd.DataFrame(arr, index = sheet.index , columns = sheet.columns)            
        return sheet


# '''
# 这一部分是可以优化的，
# 比如update 可以 只update 衍生表的几列
# 现在是全部更新 有点费时
# '''        
# class DownloadProcessTask(DownloadTask, ProcessTask):
#     pass 

    

    
    
class PV(DownloadTask):  
    
    NAME = 'PV'
    USED_DATA_LST = [wdb.EODPrices]
    DB = 'WINDDB'
    

class CDRPV(DownloadTask):
    
    NAME = 'CDRPV'
    USED_DATA_LST = [wdb.CDREODPrices]
    DB = 'WINDDB'



class PVProcess(ProcessTask):
    
    NAME = 'PVProcess'
    USED_TASK_LST = [PV, CDRPV]
    DB = 'WINDDB'
    
    def process(self):

        output_dict = {}
        task_all = (self.loadTask('PV','EODPRICES')).append(self.loadTask('CDRPV', 'CDREODPRICES'))
        
        
        trading_info = task_all[['S_INFO_WINDCODE', 'TRADE_DT', 
                                 'S_DQ_PRECLOSE', 'S_DQ_OPEN', 'S_DQ_HIGH',
                                 'S_DQ_LOW', 'S_DQ_CLOSE', 'S_DQ_VOLUME', 'S_DQ_AMOUNT',
                                 'S_DQ_TRADESTATUS', 'S_DQ_ADJFACTOR']]
        
        trading_dict = self.unstack(trading_info, 
   ['S_DQ_PRECLOSE', 'S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE', 'S_DQ_VOLUME', 'S_DQ_AMOUNT', 'S_DQ_ADJFACTOR'],
   ['TRADE_DT', 'S_INFO_WINDCODE'])
        
        
        
        
        shaped_price = task_all.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    
        suspend_sheet = shaped_price['S_DQ_TRADESTATUS'].unstack()
        suspend_sheet = suspend_sheet.replace(['交易', 'XD', 'DR', 'N', 'XR'], False)
        suspend_sheet = suspend_sheet.replace(['停牌'], True)
        
        suspend_dict = {
            'ISSUSPEND': suspend_sheet}
        
        
        
        SPZT = shaped_price['S_DQ_CLOSE'] >= shaped_price['S_DQ_LIMIT']
        SPDT = shaped_price['S_DQ_CLOSE'] <= shaped_price['S_DQ_STOPPING']
        YZZT = shaped_price['S_DQ_LOW'] >= shaped_price['S_DQ_LIMIT']
        YZDT = shaped_price['S_DQ_HIGH'] <= shaped_price['S_DQ_STOPPING']
        
        ZDT_sheet = pd.concat([SPZT, SPDT, YZZT, YZDT], axis = 1)
        ZDT_sheet.columns = ['SPZT', 'SPDT', 'YZZT', 'YZDT']
        ZDT_sheet = ZDT_sheet.sort_index(level = [0, 1])
        
        ZDT_dict = {
            'ZDT': ZDT_sheet}
        
        listedDays = (~trading_dict['S_DQ_CLOSE'].isna()).cumsum()
        listedDays = listedDays.iloc[5 * 252 :, :]
        
        listedDays_dict = {
            'LISTEDDAYS': listedDays
            }
        
        time_index = trading_dict['S_DQ_CLOSE'].index
        stock_index = trading_dict['S_DQ_CLOSE'].columns
        
        time_dict = {
            'TRADINGDAYS': pd.Series(time_index)
            }
        stock_dict = {
            'TRADINGSTOCKS': pd.Series(stock_index)
            }
        
        output_dict.update(trading_dict)
        output_dict.update(suspend_dict)
        output_dict.update(ZDT_dict)
        output_dict.update(listedDays_dict)
        output_dict.update(time_dict)
        output_dict.update(stock_dict)
        
        return output_dict

class QUARTERBAR(DownloadTask):
    '''
    这个task 不会自动load
    虽然是个download task但是他也是和其他task 有 dependency的
    
    
    
    Update的方法是错的！！
    
    万德跟天软 差了几个合并的（transfer dict）
    这个还没有改
    '''
    NAME = 'QUARTERBAR'
    USED_DATA_LST = [tsdb.QuarterMinBar]
    USED_TASK_LST = [PVProcess]
    DB = 'TSDB'
    
    save_step = '1Y'
    
    transfer_dict = {
        'SZ000022': 'SZ001872',
        'SH600849': 'SH601607',
        'SZ000043': 'SZ001914',
        'SH601313': 'SH601360'
        }
    
    
    
    @staticmethod
    def sepYears(start):
        start = pd.to_datetime(start)
        now = pd.to_datetime(str(datetime.datetime.now())[:10] )
        lst = []
        put = now + datetime.timedelta(days = 1)
        
        z = now.year - start.year
        if z == 0:
            lst.append( (start, put))
            return lst
        else:
            temp = pd.to_datetime(str(start.year + 1) )
            lst.append( (start, temp )) 
            for i in range(now.year - start.year ):
                if temp.year < now.year:
                    lst.append( (temp, pd.to_datetime(str(temp.year  + 1))) )
                    temp = pd.to_datetime(str(temp.year + 1))
                elif temp.year == now.year:
                    lst.append( (temp, put ) )
                    return lst


    @staticmethod
    def sepHalfYears(start):
        start = pd.to_datetime(start)
        now = pd.to_datetime(str(datetime.datetime.now())[:10] )
        put = now + datetime.timedelta(days = 1)
        lst = [start]
        now_year = put.year
        start_year = start.year
        for i in range(start_year, now_year):
            for j in [str(i) + '-01-01', str(i) + '-07-01']:
                j = pd.to_datetime(j)
                if j > start and j < put:
                    lst.append(j)
                if j > put:
                    break
        lst.append(put)
        res = []
        for i in range(len(lst) - 1):
            res.append( (lst[i], lst[i+1] ))
        
        return res
    @staticmethod
    def sepQuarterYears(start):
        start = pd.to_datetime(start)
        now = pd.to_datetime(str(datetime.datetime.now())[:10] )
        put = now + datetime.timedelta(days = 1)
        lst = [start]
        now_year = put.year
        start_year = start.year
        for i in range(start_year, now_year + 1):
            for j in [str(i) + '-01-01', str(i) + '-04-01', str(i) + '-07-01',
                      str(i) + '-10-01']:
                j = pd.to_datetime(j)
                if j > start and j < put:
                    lst.append(j)
                if j > put:
                    break
        lst.append(put)
        res = []
        for i in range(len(lst) - 1):
            res.append( (lst[i], lst[i+1] ))
        
        return res
                
                
    def download(self):
        
        for i in self.data_cls_dict.keys():
            stock_lst = self.loadTask('PVProcess', 'TRADINGSTOCKS')
            self.data_cls_dict[i].stock_list = stock_lst
            '''
            ts 的start , end 是含头不含尾的
            '''
            years = QUARTERBAR.sepQuarterYears(self.download_start)
            for st, ed in years:
                if st.month == 1:
                    Q = '1'
                elif st.month == 4:
                    Q = '2'
                elif st.month == 7:
                    Q = '3'
                else:
                    Q = '4'
                
                df = self.data_cls_dict[i].download(st, ed)
                df.to_pickle(
                    os.path.join(self.getDownLoadPath(), i + '_' + 'Y'+ str(st.year) + 'Q' + Q  + \
                                 '.pickle')
                    )
        
    def getUpdatedDF(self, df, init_cls):
        chosen_field = init_cls.getUpdateCalendarColumn
        start_field = df[chosen_field].max()
        start_field = pd.to_datetime(start_field) + datetime.timedelta(days = 1)
        return init_cls.update(start_field)
    
    def dropDuplicated(self, df, init_cls):
        unique_index = init_cls.unique_index
        temp = df.set_index(unique_index)        
        temp = temp[~temp.index.duplicated(keep = 'last')]
        return temp.reset_index()
    
    def updateOne(self, init_cls):
        
        
        file_name = init_cls.name
        
        thisY = datetime.datetime.now().year
        
        if pd.to_datetime(datetime.datetime.now()) < pd.to_datetime(str(thisY) + '-07-01'):
            Q = '1'
        else:
            Q = '2'
        
        df = pd.read_pickle(os.path.join(self.getDownLoadPath(),
                                file_name + '_' + 'Y'+ str(thisY) + 'Q' + Q + '.pickle'))
        
        res =  self.dropDuplicated(df.append(self.getUpdatedDF(df, init_cls)), init_cls)
        
        self.updated_records = len(res) - len(df)
        return res
        

    def update(self):
        thisY = datetime.datetime.now().year
        for i in self.data_cls_dict.keys():
            stock_lst = self.loadTask('PVProcess', 'TRADINGSTOCKS')
            self.data_cls_dict[i].stock_lst = stock_lst
            
            df = self.updateOne(self.data_cls_dict[i])
            file_name = self.data_cls_dict[i].name
            df.to_pickle(
                os.path.join(self.getDownLoadPath(), 
                             file_name + '_' + 'Y'+ str(thisY) + '.pickle')
                )
    
    
    
    
    
    
class Financial(DownloadTask):
    download_start = '20050101'
    NAME = 'FINANCIAL'
    USED_DATA_LST = [wdb.BalanceSheet, wdb.IncomeStatement, wdb.CashflowStatement]
    DB = 'WINDDB'

    
class IndexEOD(DownloadTask):
    NAME = 'INDEX'
    USED_DATA_LST = [wdb.IndexEOD] 
    DB = 'WINDDB'


class IndexEODProcess(ProcessTask):
    NAME = 'INDEXProcess'
    USED_TASK_LST = [IndexEOD]
    DB = 'WINDDB'
    
    def process(self):
        df = self.loadTask('INDEX', 'INDEXEOD')
        res = df.set_index(['S_INFO_WINDCODE', 'TRADE_DT']).sort_index(level = [0, 1])
        
        res_dict = {}
        for i in res.index.levels[0]:
            t = res.loc[i]
            t.index = pd.to_datetime(t.index.astype(str))
            res_dict[i] = t
        return res_dict
    
class FreeFloat(DownloadTask):
    download_start = '20000101'
    NAME = 'FREEFLOAT'
    USED_DATA_LST = [wdb.FreeFloat]
    DB = 'WINDDB'

class IndexWeight(DownloadTask):
    NAME = 'INDEXWEIGHT'
    USED_DATA_LST = [tsdb.IndexWeight]
    DB = 'TSDB'
    

class IndexWeightProcess(ProcessTask):
    NAME = 'INDEXWEIGHTProcess'
    USED_TASK_LST = [IndexWeight]
    DB = 'TSDB'
    
    def process(self):
        res = self.loadTask('INDEXWEIGHT', 'INDEXWEIGHT').set_index(
            ['指数代码','截止日', '代码'])['比例(%)'].sort_index(
                level = [0,1,2]).unstack()
        res.columns = [i[2:] + '.' + i[:2] for i in res.columns]
        
        res_dict = {}
        for i in res.index.levels[0]:
            temp = res.loc[i]
            temp = temp.reset_index()
            temp['截止日'] = temp['截止日'].astype(str)
            temp = temp.set_index(['截止日'])
            temp.index = pd.to_datetime(temp.index)
            res_dict[i] = temp
        return res_dict


class MarketCap(DownloadTask):
    download_start = '20000101'
    NAME = 'MARKETCAP'
    USED_DATA_LST = [wdb.MarketCap]
    DB = 'WINDDB'


class CDRMarketCap(DownloadTask):
    download_start = '20050101'
    NAME = 'CDRMARKETCAP'
    USED_DATA_LST = [wdb.CDRMarketCap]
    DB = 'WINDDB'



class ST(DownloadTask):
    download_start = '19900101'
    NAME = 'ST'
    USED_DATA_LST = [wdb.ST]
    DB = 'WINDDB'
    

class SW1IND(DownloadTask):
    download_start = '19900101'
    NAME = 'SW1IND_OLD'
    USED_DATA_LST = [wdb.SWINDUSTRYCOMPONENT_OLD]
    DB = 'WINDDB'    

class SW1IND_NEW(DownloadTask):
    download_start = '19900101'
    NAME = 'SW1IND_NEW'
    USED_DATA_LST = [wdb.SWINDUSTRYCOMPONENT_NEW]
    DB = 'WINDDB'    


    
class STSheet(ProcessTask):
    NAME = 'STSHEET'
    DB = 'MIXED'
    USED_TASK_LST = [PVProcess, ST]
    

    def process(self):
        stocks = self.loadTask('PVProcess', 'TRADINGSTOCKS').values
        dates = self.loadTask('PVProcess', 'TRADINGDAYS').values
        
        template = pd.DataFrame(np.nan, index = dates, columns = stocks)
        
        st = self.loadTask('ST', 'ST')
        
        
        st_sheet = self.fillSheetByEntryRemove(template, st[(st['S_TYPE_ST'] == 'S') 
                                      |(st['S_TYPE_ST'] == 'X') 
                                      |(st['S_TYPE_ST'] == 'L') ].copy())
        exit_sheet =  self.fillSheetByEntryRemove(template, st[(st['S_TYPE_ST'] == 'T')].copy())
        return {
            'ST':st_sheet.fillna(False),
            'EXIT': exit_sheet.fillna(False)
            }

class IndustryDictionary(DownloadTask):
    download_start = '19900101'
    NAME = 'INDUSTRY_DICTIONARY'
    USED_DATA_LST = [wdb.IndustryDict]
    DB = 'WINDDB'
    
        
class Caps(ProcessTask):
    NAME = 'CAPS'
    DB = 'MIXED'
    USED_TASK_LST = [PVProcess, MarketCap, FreeFloat, CDRMarketCap]
    
    def process(self):
        
        stocks = self.loadTask('PVProcess', 'TRADINGSTOCKS').values
        dates = self.loadTask('PVProcess', 'TRADINGDAYS').values
        
        caps = self.loadTask('MARKETCAP', 'MARKETCAP')
        CDRcaps = self.loadTask('CDRMARKETCAP', 'CDRMARKETCAP')
        caps = caps.append(CDRcaps)
        
        caps = caps[caps['IS_VALID'] == 1]
    
        
        
        
        
        caps.loc[:, 'date'] = caps.loc[:, ['CHANGE_DT', 'ANN_DT']].max(axis = 1).astype(int).astype(str)        
        caps = caps.sort_values(
            ['WIND_CODE', 'date', 'CHANGE_DT', 'CHANGE_DT1', 'OPDATE'])
        
        caps.loc[:, 'cummax_date'] = caps.groupby('WIND_CODE')['CHANGE_DT'].apply(lambda x: x.cummax())
        caps = caps[~caps.set_index(['WIND_CODE', 'cummax_date']).index.duplicated(
            keep = 'first')].reset_index().set_index('index')
        
        res =  self.unstack(
            caps, 
            ['TOT_SHR', 'FLOAT_SHR', 'S_SHARE_NTRD_PRFSHARE'],
            ['date', 'WIND_CODE'])
        
        
        freecaps = self.loadTask('FREEFLOAT', 'FREEFLOAT')

        

        freecaps.loc[:, 'date'] = freecaps.loc[:, ['CHANGE_DT', 'ANN_DT']].max(axis = 1).astype(int).astype(str)
        freecaps = freecaps.sort_values(
            ['S_INFO_WINDCODE', 'date', 'CHANGE_DT', 'CHANGE_DT1', 'OPDATE'])
        
        freecaps.loc[:,'cummax_date'] = freecaps.groupby('S_INFO_WINDCODE')['CHANGE_DT'].apply(lambda x: x.cummax())
        freecaps = freecaps[~freecaps.set_index(['S_INFO_WINDCODE', 'cummax_date']).index.duplicated(
            keep = 'first')].reset_index().set_index('index')
        
        
        freecap_res = self.unstack(
            freecaps,
            ['S_SHARE_FREESHARES'],
            ['date', 'S_INFO_WINDCODE']
            
            )
        for i in res.keys():
            temp = res[i]
            temp.index = pd.to_datetime(temp.index.astype(str))
            temp.loc[:, list(set(stocks) - set(temp.columns))] = np.nan
            temp = temp.loc[:, stocks]
            temp = temp[:dates[-1]]
            temp.index = dates[np.searchsorted(dates, temp.index)]
            temp = temp.fillna(method = 'ffill')
            temp = temp[~temp.index.duplicated(keep = 'last')]
            if len(temp.index) != len(dates):
                temp = pd.concat([temp, pd.Series(0, index = dates)], axis = 1).iloc[:, :-1]

            res[i] = temp
            
        for i in freecap_res.keys():
            temp = freecap_res[i]
            temp.index = pd.to_datetime(temp.index.astype(str))
            temp.loc[:, list(set(stocks) - set(temp.columns))] = np.nan
            temp = temp.loc[:, stocks]
            temp = temp[:dates[-1]]
            temp.index = dates[np.searchsorted(dates, temp.index)]
            temp = temp.fillna(method = 'ffill')
            temp = temp[~temp.index.duplicated(keep = 'last')]
            if len(temp.index) != len(dates):
                temp = pd.concat([temp, pd.Series(0, index = dates)], axis = 1).iloc[:, :-1]
            
            freecap_res[i] = temp    
        
        res.update(freecap_res)
        return res

class SW1INDMERGED(ProcessTask):
    NAME = 'SW1IND'
    DB = 'MIXED'
    USED_TASK_LST = [PVProcess, SW1IND, SW1IND_NEW, IndustryDictionary]
    @staticmethod
    def getLevelDict(df, option):
        if option == 'old':
            used = df[[x.startswith('61') for x in df['INDUSTRIESCODE']]]
        else:
            used = df[[x.startswith('76') for x in df['INDUSTRIESCODE']]]
        level1 = used[used['LEVELNUM'] == 2]
        level1.loc[:, 'INDUSTRIESCODE_OLD'] = [x[:4] for x in level1['INDUSTRIESCODE_OLD']]
        level1 = level1.set_index('INDUSTRIESCODE_OLD')
        level2 = used[used['LEVELNUM'] == 3]
        level2.loc[:, 'INDUSTRIESCODE_OLD'] = [x[:6] for x in level2['INDUSTRIESCODE_OLD']]
        level2 = level2.set_index('INDUSTRIESCODE_OLD')
        if option == 'old':
            level1 = level1.append(level2[
                [('银行' in x) | ('证券' in x) | ('保险' in x) | ('多元金融' in x)  
                                            for x in level2['INDUSTRIESNAME']]])
        else:
            level1 = level1.append(level2[[('证券' in x) | ('保险' in x) | ('多元金融' in x) 
                                            for x in level2['INDUSTRIESNAME']]])
        level1.loc[:, 'INDUSTRIESNAME'] = [i.replace('Ⅱ', '')
            for i in level1['INDUSTRIESNAME']]
        
        return level1['INDUSTRIESNAME']
    
    
    @staticmethod
    def giveIndustryClassificationFunction(df):
        old_level1 = SW1INDMERGED.getLevelDict(df, 'old')
        new_level1 = SW1INDMERGED.getLevelDict(df, 'new')
        ind_name = (old_level1.append(new_level1)).unique()
        ind_name_dic = dict(zip(ind_name,
                                ['SW' + str(i+1) for i in range(len(ind_name))]))
        old_level1 = old_level1.map(ind_name_dic)
        new_level1 = new_level1.map(ind_name_dic)
        
        def func(sw_code,option):
            if option == 'old':
                try:
                    return old_level1.loc[sw_code[:4]]
                except:
                    return old_level1.loc[sw_code[:6]]
            elif option == 'new':
                try:
                    return new_level1.loc[sw_code[:4]]
                except:
                    return new_level1.loc[sw_code[:6]]
        
        return func, ind_name_dic
 
    def process(self):
        
        stocks = self.loadTask('PVProcess', 'TRADINGSTOCKS').values
        dates = self.loadTask('PVProcess', 'TRADINGDAYS').values
        
        template = pd.DataFrame(np.nan, index = dates, columns = stocks)
        
        
        old_sheet = self.loadTask('SW1IND_OLD', 'SWINDUSTRYCOMPONENT')
        new_sheet = self.loadTask('SW1IND_NEW', 'SWINDUSTRYCOMPONENT_NEW')
        
        dictionary = self.loadTask('INDUSTRY_DICTIONARY', 'INDUSTRYDICTIONARY')
        
        
        myclsfunc, ind_dic = SW1INDMERGED.giveIndustryClassificationFunction(
            dictionary
            )

        old_sheet.loc[:, 'SW_IND_CODE'] = old_sheet.loc[:, 'SW_IND_CODE'].apply(lambda x:
                                                                  myclsfunc(x, option = 'old'))
        
        new_sheet.loc[:, 'SW_IND_CODE'] = new_sheet.loc[:, 'SW_IND_CODE'].apply(lambda x:
                                                                  myclsfunc(x, option = 'new'))
                
        old = self.fillSheetByEntryRemove(template, old_sheet, val_column = 'SW_IND_CODE')
        new = self.fillSheetByEntryRemove(template, new_sheet, val_column = 'SW_IND_CODE')
        
        old = old.fillna(method = 'ffill')
        new = new.fillna(method = 'ffill')
        res = old[:'2021-12-12'].append(new['2021-12-13':])
        return {'MYIND': res.fillna(method = 'ffill'),
                'IND_DICT': pd.Series(ind_dic)}
        




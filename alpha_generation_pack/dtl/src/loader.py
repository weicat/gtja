# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 09:32:10 2021

@author: Administrator
"""
import pandas as pd
import os
import numpy as np 
from itertools import chain
from functools import reduce, wraps
import collections
import dtl.src.data_utils as dtls
import numba
from tqdm import tqdm




    

def reform(func):
    @wraps(func)
    def wrapper(self, *arg, **kw):
        
        res = func(self, *arg, **kw)
        temp = res.index[0]
        if not isinstance(temp, pd.Timestamp):
            if isinstance(temp, int):
                res.index = res.index.astype(str)
            res.index = pd.to_datetime(res.index)
        return res
    return wrapper


def save(name):
    def Inner(func):
        def wrapper(self, *arg, **kw):
            if hasattr(self, name):
                return eval('self.{t}'.format(t = name))
            res = func(self, *arg, **kw)
            exec('self.{f} = res'.format(f = name))
            return res
        return wrapper
    return Inner

class DataLoader(object):
    
    def __init__(self, database_path = r'Z:\LuTianhao\DataBase'):
        self.database_path = database_path
        self.configuration()
    

    @staticmethod
    def _align(df1, df2, *dfs):
        dfs_all = [df for df in chain([df1, df2], dfs)]
        if any(len(df.shape) == 1 or 1 in df.shape for df in dfs_all):
            dims = 1
        else:
            dims = 2
        mut_date_range = sorted(reduce(lambda x,y: x.intersection(y), (df.index for df in dfs_all)))
        mut_codes = sorted(reduce(lambda x,y: x.intersection(y), (df.columns for df in dfs_all)))
        if dims == 2:
            dfs_all = [df.loc[mut_date_range, mut_codes] for df in dfs_all]
        elif dims == 1:
            dfs_all = [df.loc[mut_date_range, :] for df in dfs_all]
        return dfs_all
    
    
    def updateNestedDictionary(self, existing, new):
    
        for k, v in new.items():
            if isinstance(existing, collections.Mapping):
                if isinstance(v, collections.Mapping):
                    r = self.updateNestedDictionary(existing.get(k, {}), v)
                    existing[k] = r
                else:
                    existing[k] = new[k]
            else:
                existing = {k: new[k]}
        return existing 
        
    
    def createNestedDictionary(self, lst, abspath):
        
        if len(lst) == 1:
            file_name = lst[0]
            return {'.'.join(file_name.split('.')[:-1]) : abspath}
        else:
            folder = lst[0]
            lst = lst[1:]
            return {folder: self.createNestedDictionary(lst, abspath)}
        
    
    
    def configuration(self):
        all_file_paths = []
        for (dirpath, dirnames, filenames) in os.walk(self.database_path):
            for i in filenames:
                all_file_paths.append(os.path.join(dirpath, i))
        used = [f.replace(os.path.join(self.database_path, ''), '') 
                for f in all_file_paths]
        self.configured_dict = {}
        
        for abs_path, p in zip(all_file_paths, used):
            folder_and_files = p.split('\\')
            d = self.createNestedDictionary(folder_and_files, abs_path)
            self.configured_dict = self.updateNestedDictionary(self.configured_dict, d)
        
    
    def read(self, *args):
        t = eval("self.configured_dict" + "[\"" + "\"][\"".join(args) + "\"]" )
        
        if isinstance(t, str):
            exec("self.configured_dict" + "[\"" + "\"][\"".join(args) + "\"] = pd.read_pickle(r\"{t}\")".format(t = t) ) 
            return eval("self.configured_dict" + "[\"" + "\"][\"".join(args) + "\"]" )
        else:
            return t
    
    def setVal(self, val, *args):
        
        exec("self.configured_dict" + "[\"" + "\"][\"".join(args) + "\"] = val" ) 
        
    
    
    
    def delete(self, *args):
        exec("self.configured_dict" + "[\"" + "\"][\"".join(args) + "\"] = [] " ) 

    

    def unstack(self, df):
        '''
        默认第一层是时间
        '''
        time = self.getTradeDays()
        stock = self.getTradeStocks()
        time_name, stock_name = df.index.names
        df = df.reset_index()
        df = df[(df[time_name] <= time[-1]) & (df[stock_name].isin(stock)) ]
        df.loc[:, time_name] = time[np.searchsorted(time, df[time_name])]
        temp = df.set_index([stock_name, time_name]).iloc[:,0]
        temp = temp[~temp.index.duplicated(keep = 'last')].unstack(level = 0)
        temp[list(set(stock) - set(temp.columns))] = np.nan
        temp = pd.concat([temp, pd.Series(0, index = time)], axis = 1).iloc[:, :-1]
        return temp
        # df = df.sort_index(level = [0,1])
        # df = df[:time[-1]]
        # df.index.levels[0] = df.index.levels[0]    
    
    def getIndexWeight(self, index_name):
        t = index_name.split('.')
        return self.read('TSDB','calculated','INDEXWEIGHTProcess', t[1] + t[0])
        
    
    def getSuspend(self):
        df = self.read('WINDDB', 'calculated', 'PVProcess', 'ISSUSPEND')
        return df
    
    def getRealizedNextPeriodReturn(self):
        exit_ = self.getExit()
        exitMarketIndex = exit_.iloc[-1,:][exit_.iloc[-1,:] == True].index
        realizedReturn = self.getStockRet().copy()
        
        exit_ = exit_.replace([True], np.nan)
        for exitStock in exitMarketIndex:
            exitPos = exit_[exitStock].index.get_loc(exit_[exitStock].last_valid_index())
            if exitPos:
                if exitPos < len(exit_) - 1:
                    exitTime = exit_.index[exitPos + 1] 
                    realizedReturn.loc[exitTime, exitStock] = -1
        realizedReturn[exit_ == 1] = 0
        self.realizedNextPeriodReturn = realizedReturn.shift(-1)
        return self.realizedNextPeriodReturn
    
   
    def getST(self):
        df = self.read('MIXED', 'calculated', 'STSHEET', 'ST')
        return df
    
    def getExit(self):
        df = self.read('MIXED', 'calculated', 'STSHEET', 'EXIT')
        return df
    
    @save(name = 'time')    
    def getTradeDays(self):
        df = self.read('WINDDB', 'calculated', 'PVProcess', 'TRADINGDAYS')
        return df.values
    
    @save(name = 'stocks')    
    def getTradeStocks(self):
        df = self.read('WINDDB', 'calculated', 'PVProcess', 'TRADINGSTOCKS')
        return df.values
    
    @reform
    @save(name = 'stock_ret')
    def getStockRet(self):
        df = self.getAdjClose()
        suspend = self.getSuspend().fillna(False)
        df = df.pct_change()
        suspend, df = DataLoader._align(suspend, df)
        df[suspend] = np.nan
        return df
    
    @save(name = 'adjclose')
    def getAdjClose(self):
        df = self.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_CLOSE')
        adj = self.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_ADJFACTOR')
        df = df * adj
        return df
    
    
    @reform
    @save(name = 'list_days')
    def getListedDays(self):
        df = self.read('WINDDB', 'calculated', 'PVProcess', 'LISTEDDAYS')
        return df
    
    # @save(name = 'spzt')
    # def getSPZT(self):
    #     df = pd.read_pickle(r'Z:\barra_V1\rawData\basics_total\SPZT.pickle')
    #     lst = []
    #     for i in df.columns:
    #         market = i[:2]
    #         code = i[2:]
    #         if market == 'NE':
    #             market = 'BJ'
    #         lst.append(code + '.' + market)
    #     df.columns = lst
    #     df = df.sort_index(axis = 1)
    #     return df

    # @save(name = 'spdt')
    # def getSPDT(self):
    #     df = pd.read_pickle(r'Z:\barra_V1\rawData\basics_total\SPDT.pickle')
    #     lst = []
    #     for i in df.columns:
    #         market = i[:2]
    #         code = i[2:]
    #         if market == 'NE':
    #             market = 'BJ'
    #         lst.append(code + '.' + market)
    #     df.columns = lst
    #     df = df.sort_index(axis = 1)
    #     return df

    @reform
    @save(name = 'spzt')
    def getSPZT(self):
        df = self.read('WINDDB', 'calculated', 'PVProcess', 'ZDT')
        return df['SPZT'].unstack()
    
    @reform
    @save(name = 'spdt')
    def getSPDT(self):
        df = self.read('WINDDB', 'calculated', 'PVProcess', 'ZDT')
        return df['SPDT'].unstack()
    
    @reform
    @save(name = 'stock_ret_2')
    def getStockRetBoundedAndDropFirstFive(self):
        ret = self.getStockRet()
        listdays = self.getListedDays()
        c = ret.loc[listdays.index, listdays.columns].copy()
        c[listdays <= 5] = np.nan
        ret.loc[listdays.index, listdays.columns] = c
        return ret.clip(-0.2, 0.2)
    
    @reform
    # @save(name = 'mkt_ret')
    def getMarketInfo(self, field = 'S_DQ_CLOSE', 
                      market_index = '000001.SH'):
        df = self.read('WINDDB', 'raw', 'INDEX', 'INDEXEOD' )
        index = df[df['S_INFO_WINDCODE'] == market_index]      
        df = index.set_index('TRADE_DT')[field]
        df = df.sort_index()
        df.index = pd.to_datetime(df.index)
        return df
    @reform
    # @save(name = 'mkt_ret')
    def getMarketRet(self, market_index = '000001.SH'):
        df = self.read('WINDDB', 'raw', 'INDEX', 'INDEXEOD' )
        index = df[df['S_INFO_WINDCODE'] == market_index]      
        df = index.set_index('TRADE_DT')['S_DQ_CLOSE']
        df = df.sort_index()
        df = df.pct_change().dropna().astype(float)
        df.index = pd.to_datetime(df.index)
        return df
    
    @save(name = 'PE')
    def getPriorSharesMV(self):
        '''
        优先股是万股
        '''
        df = self.read('MIXED', 'calculated', 'CAPS', 'S_SHARE_NTRD_PRFSHARE')
        close = self.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_CLOSE')
        res = df.astype(float) * close * 1e4
        return res
    
    @save(name = 'MV')
    def getMarketValue(self):
        
        shares = self.read('MIXED', 'calculated', 'CAPS', 'TOT_SHR')
        close = self.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_CLOSE')
        res = shares.astype(float) * close * 1e4
        return res.fillna(method = 'ffill')
    
    def getTurnOver(self):
        
        floatshares = self.read('MIXED', 'calculated', 'CAPS', 'FLOAT_SHR').astype(float)
        volume = self.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_VOLUME')
        return (volume/floatshares)* 1e-2
        
    
    '''
    这里还能加快 合并相同的financial utils
    '''
    def getBookValue(self):
        df = self.read('WINDDB', 'raw', 'FINANCIAL','BALANCESHEET')
        if isinstance(df, dtls.FinancialDataUtils):
            futils = df
        else:
            futils = dtls.FinancialDataUtils(df)
            self.setVal(futils, 'WINDDB', 'raw', 'FINANCIAL','BALANCESHEET')
        futils.method = 'fill'
        res = futils.nearest('TOT_SHRHLDR_EQY_INCL_MIN_INT')
        return self.unstack(res).fillna(method = 'ffill').astype(float)
    
    def getTotalAsset(self):
        df = self.read('WINDDB', 'raw', 'FINANCIAL','BALANCESHEET')
        if isinstance(df, dtls.FinancialDataUtils):
            futils = df
        else:
            futils = dtls.FinancialDataUtils(df)
            self.setVal(futils, 'WINDDB', 'raw', 'FINANCIAL','BALANCESHEET')
        futils.method = 'fill'
        res = futils.nearest('TOT_ASSETS')
        return self.unstack(res).fillna(method = 'ffill').astype(float)
    
    def getTotalLiability(self):
        df = self.read('WINDDB', 'raw', 'FINANCIAL','BALANCESHEET')
        if isinstance(df, dtls.FinancialDataUtils):
            futils = df
        else:
            futils = dtls.FinancialDataUtils(df)
            self.setVal(futils, 'WINDDB', 'raw', 'FINANCIAL','BALANCESHEET')
        futils.method = 'fill'
        res = futils.nearest('TOT_LIAB')
        return self.unstack(res).fillna(method = 'ffill').astype(float)        
    
    @save('ciruclate_marketvalue')
    def getCirculateMarketValue(self):
        df = self.read('MIXED', 'calculated', 'CAPS', 'FLOAT_SHR').astype(float)
        close = self.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_CLOSE').astype(float)
        return df*close
    
    
    @save('trailing_netprofit')
    def getTrailingNetProfit(self):
        df = self.read('WINDDB', 'raw', 'FINANCIAL','INCOMESTATEMENT')
        if isinstance(df, dtls.FinancialDataUtils):
            futils = df
        else:
            futils = dtls.FinancialDataUtils(df)
            self.setVal(futils, 'WINDDB', 'raw', 'FINANCIAL','INCOMESTATEMENT')
    
        futils.method = 'ignore'
        res = futils.rolling(chosen_field = 'NET_PROFIT_INCL_MIN_INT_INC', 
                       return_obj = True)
        futils.reset()
        return res
    
    @save('trailing_cash_netprofit')
    def getTrailingCashNetProfit(self):
        df = self.read('WINDDB', 'raw', 'FINANCIAL','CASHFLOWSTATEMENT')
        if isinstance(df, dtls.FinancialDataUtils):
            futils = df
        else:
            futils = dtls.FinancialDataUtils(df)
            self.setVal(futils, 'WINDDB', 'raw', 'FINANCIAL','CASHFLOWSTATEMENT')
    
        futils.method = 'ignore'
        res = futils.rolling(chosen_field = 'NET_INCR_CASH_CASH_EQU', 
                       return_obj = True)
        futils.reset()
        return res
        
    
    
    
    @save('trailing_operrev')
    def getTrailingOperRevenue(self):
        df = self.read('WINDDB', 'raw', 'FINANCIAL','INCOMESTATEMENT')
        if isinstance(df, dtls.FinancialDataUtils):
            futils = df
        else:
            futils = dtls.FinancialDataUtils(df)
            self.setVal(futils, 'WINDDB', 'raw', 'FINANCIAL','INCOMESTATEMENT')
    
        futils.method = 'ignore'
        res = futils.rolling(chosen_field = 'OPER_REV', 
                       return_obj = True)
        futils.reset()
        return res    
    
    def getNonCurLiability(self):
        df = self.read('WINDDB', 'raw', 'FINANCIAL', 'BALANCESHEET')
        if isinstance(df, dtls.FinancialDataUtils):
            futils = df
        else:
            futils = dtls.FinancialDataUtils(df)
            self.setVal(futils, 'WINDDB', 'raw', 'FINANCIAL','BALANCESHEET')
        futils.method = 'ignore'
        res = futils.nearest('TOT_LIAB')
        res = futils.nearest('TOT_NON_CUR_LIAB')
        return self.unstack(res).fillna(method = 'ffill')
    
    def getPriorSharesDebt(self):
        df = self.read('WINDDB', 'raw', 'FINANCIAL', 'BALANCESHEET')
        if isinstance(df, dtls.FinancialDataUtils):
            futils = df
        else:
            futils = dtls.FinancialDataUtils(df)
            self.setVal(futils, 'WINDDB', 'raw', 'FINANCIAL','BALANCESHEET')
        futils.method = 'fill'
        res = futils.nearest('OTHER_EQUITY_TOOLS_P_SHR')
        return self.unstack(res).fillna(method = 'ffill')
    
    @save('Industry')
    def getIndustry(self):
        df = self.read('MIXED','calculated', 'SW1IND', 'MYIND')
        return df 
    
    
    
    def getQuarterMinData(self, Y_start, Q_start, Y_end, Q_end):
        
        Y_start = int(Y_start)
        Q_start = int(Q_start)
        Y_end = int(Y_end)
        Q_end = int(Q_end)
        
        lst = []
        z = Q_start
        for y in tqdm(range(Y_start, Y_end + 1), desc = 'LoadingData'):
            if y == Y_end:
                tz = Q_end + 1
            else:
                tz = 5
                
            for q in range(z, tz):
                lst.append(
                    pd.read_pickle(
                        os.path.join(
                            r'Z:\LuTianhao\DataBase\TSDB\raw\QUARTERBAR',
                            'QUARTERMINBAR_Y{y}Q{q}.pickle'.format(y = y,
                                                                   q = q)
                            )
                        )
                    
                    )
            z = 1
        return lst
        
        
    
    
    # @save('Industry')
    # def getIndustry(self):
    #     df = pd.read_pickle(r'Z:\barra_V1\rawData\basics_total\SWIndustryID1Adj.pickle')
    #     return df 
        
    
class BarraLoader(DataLoader):
    style_factor_name = ['Beta', 
                         'BooktoPrice',
                         'EarningsYield',
                         'Growth',
                         'Leverage', 
                         'Liquidity',
                         'Momentum',
                         'NonLinearSize',
                         'ResidualVolatility',
                         'Size']
    def getFinalFactor(self, name):
        t = self.read('Cal_factor', name)
        if len(t.columns[0].split('.')) == 1:
            t.columns = [i[2:] + '.' + i[:2]for i in t.columns]
        
        return t

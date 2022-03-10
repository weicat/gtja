# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 08:53:07 2022

@author: Administrator
"""

import pandas as pd
from collections import Counter

def qualitycheck(df):
    try:    
        temp = [i.date() for i in df]
    except:
        temp = [i for i in df]
    
    temp = pd.Series(dict(Counter(temp)))
    temp.index = pd.to_datetime(temp.index)
    return temp.sort_index()


def statementTypeCheck(df):
    
    # df = df[(df['STATEMENT_TYPE']!= '408006000') & (df['STATEMENT_TYPE']!= '408007000' )]
    check = df.sort_values(
        by = ['REPORT_PERIOD', 'ACTUAL_ANN_DT', 'STATEMENT_TYPE']
        ).groupby(['WIND_CODE', 'REPORT_PERIOD'])['STATEMENT_TYPE'].unique()
    
    chosen_index = []
    for i in check.index.levels[0]:
        if (i.split('.')[1]!='SZ' and i.split('.')[1]!='SH') or i[0]=='A':
            continue
        chosen_index.append(i)
    
    check = check.loc[chosen_index]    
    return check


def getTradingDays(conn, start = '2010-01-01'):
    
    time = pd.to_datetime(start)
    time = ''.join((str(time)[:10]).split('-'))
    sql_string = '''
            SELECT TRADE_DAYS 
            FROM [dbo].[ASHARECALENDAR] 
            WHERE S_INFO_EXCHMARKET = 'SSE'
            AND TRADE_DAYS >= N'{t}'
            
        '''.format(t = time)
    cursor = conn.conn.cursor()
    cursor.execute(sql_string)
    res = cursor.fetchall()
    df = pd.DataFrame( res, columns = [i[0] for i in cursor.description])
    df = df.iloc[:,0].sort_values()
    cursor.close()
    return df


    
    

class Data(object):
    '''
    所有的update是 >= 
    可能会重复，在每日update的时候会drop duplicate 相应的index
    '''
    def __init__(self, db):
        self.conn = db.conn
        
    def toDataFrame(self, df ):
        return pd.DataFrame( df, 
                            columns = [i[0] for i in self.cursor.description])
    
    def define_mysql_order(self):
        self.sql_order = None
        raise NotImplementedError("need to have sql_order")
    
    def update_mysql_order(self, begin_time):
        
        self.sql_order = self.sql_order + ' WHERE {calendar_column} >= N\'{f}\''.format(
            f = begin_time, calendar_column = self.getUpdateCalendarColumn)
    
    
    @property 
    def name(self):
        return self.name    
            
    @property    
    def getCalendarColumn(self):
        raise NotImplementedError("need to have calendar_column")

    @property    
    def getUpdateCalendarColumn(self):
        raise NotImplementedError("need to have update_calendar_column")

    
    def download_mysql_order(self, begin_time):
        
        if self.getCalendarColumn != '':
            self.sql_order = self.sql_order + ' WHERE {calendar_column} >= N\'{f}\''.format(
                f = begin_time, calendar_column = self.getCalendarColumn)
        
    

    def fetchall(self):
        self.cursor.execute(self.sql_order)
        return self.cursor.fetchall()
    
    
    def process(self, df):
        return df
    
    def download(self, begin_time):
        self.define_mysql_order()
        self.download_mysql_order(begin_time)
        self.cursor = self.conn.cursor()
        temp = self.fetchall()
        temp = self.toDataFrame(temp)
        final = self.process(temp)
        self.cursor.close()
        return final
    
    
    def update(self, begin_time):
        #begin time with format yyyy-mm-dd hh:mm:ss
        
        self.define_mysql_order()
        self.update_mysql_order(begin_time)
        self.cursor = self.conn.cursor()
        temp = self.fetchall()
        temp = self.toDataFrame(temp)
        final = self.process(temp)
        self.cursor.close()
        return final

    
    
        
class EODPrices(Data):
    unique_index = ['TRADE_DT','S_INFO_WINDCODE']
    name = 'EODPRICES'
    
    
    @property    
    def getCalendarColumn(self):
        return 'TRADE_DT'
    
    @property    
    def getUpdateCalendarColumn(self):
        return 'TRADE_DT'

    
    def define_mysql_order(self):
        self.sql_order = '''
        SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_PRECLOSE, S_DQ_OPEN, S_DQ_HIGH, 
        S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME, S_DQ_AMOUNT, S_DQ_TRADESTATUS, 
        S_DQ_ADJFACTOR, S_DQ_LIMIT, S_DQ_STOPPING
        FROM [dbo].[AShareEODPrices] 
        '''
    
    def process(self, df):
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
        df[['S_DQ_PRECLOSE', 'S_DQ_OPEN', 'S_DQ_HIGH',
        'S_DQ_LOW', 'S_DQ_CLOSE', 'S_DQ_VOLUME', 'S_DQ_AMOUNT', 
        'S_DQ_ADJFACTOR', 'S_DQ_LIMIT', 'S_DQ_STOPPING']] = df[['S_DQ_PRECLOSE', 'S_DQ_OPEN', 'S_DQ_HIGH',
        'S_DQ_LOW', 'S_DQ_CLOSE', 'S_DQ_VOLUME', 'S_DQ_AMOUNT', 
        'S_DQ_ADJFACTOR', 'S_DQ_LIMIT', 'S_DQ_STOPPING']].astype(float)
        df = df.sort_values(by = 'TRADE_DT')
        
        return df
        
class CDREODPrices(Data):
    unique_index = ['TRADE_DT','S_INFO_WINDCODE']
    name = 'CDREODPRICES'
    @property    
    def getCalendarColumn(self):
        return 'TRADE_DT'
    
    @property    
    def getUpdateCalendarColumn(self):
        return 'TRADE_DT'

    
    def define_mysql_order(self):
        self.sql_order = '''
        SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_PRECLOSE, S_DQ_OPEN, S_DQ_HIGH, 
        S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME, S_DQ_AMOUNT, S_DQ_TRADESTATUS, 
        S_DQ_ADJFACTOR, S_DQ_LIMIT, S_DQ_STOPPING
        FROM [dbo].[CDREODPrices] 
        '''
    
    def process(self, df):
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
        df[['S_DQ_PRECLOSE', 'S_DQ_OPEN', 'S_DQ_HIGH',
        'S_DQ_LOW', 'S_DQ_CLOSE', 'S_DQ_VOLUME', 'S_DQ_AMOUNT', 
        'S_DQ_ADJFACTOR', 'S_DQ_LIMIT', 'S_DQ_STOPPING']] = df[['S_DQ_PRECLOSE', 'S_DQ_OPEN', 'S_DQ_HIGH',
        'S_DQ_LOW', 'S_DQ_CLOSE', 'S_DQ_VOLUME', 'S_DQ_AMOUNT', 
        'S_DQ_ADJFACTOR', 'S_DQ_LIMIT', 'S_DQ_STOPPING']].astype(float)
        df = df.sort_values(by = 'TRADE_DT')
        
        return df


class FreeFloat(Data):
    unique_index = ['S_INFO_WINDCODE', 'CHANGE_DT1']
    name = 'FREEFLOAT'
    
    @property    
    def getCalendarColumn(self):
        return 'CHANGE_DT1'
    
    @property    
    def getUpdateCalendarColumn(self):
        return 'OPDATE'
    
    def define_mysql_order(self):
        self.sql_order = '''
        SELECT S_INFO_WINDCODE, CHANGE_DT, S_SHARE_FREESHARES,
               CHANGE_DT1, ANN_DT, OPDATE
        FROM [dbo].[AShareFreeFloat] 
        '''
        
class FinancialData(Data):
    
    @property    
    def getCalendarColumn(self):
        return 'REPORT_PERIOD'
    
    @property    
    def getUpdateCalendarColumn(self):
        return 'OPDATE'
    
    
    
class BalanceSheet(FinancialData):
    unique_index = ['WIND_CODE', 'ACTUAL_ANN_DT', 'REPORT_PERIOD',
                    'STATEMENT_TYPE', 'OPDATE']
    name = 'BALANCESHEET'


    

    def define_mysql_order(self):

        self.sql_order = '''
        SELECT WIND_CODE, ANN_DT, REPORT_PERIOD, STATEMENT_TYPE, TOT_CUR_ASSETS,
        TOT_ASSETS, TOT_NON_CUR_LIAB, TOT_LIAB,
        TOT_SHRHLDR_EQY_INCL_MIN_INT,OTHER_EQUITY_TOOLS_P_SHR,
        ACTUAL_ANN_DT, OPDATE
        
        FROM [dbo].[ASHAREBALANCESHEET]
        '''


class IncomeStatement(FinancialData):
    unique_index = ['WIND_CODE', 'ACTUAL_ANN_DT', 'REPORT_PERIOD',
                    'STATEMENT_TYPE', 'OPDATE']
    name = 'INCOMESTATEMENT'


    def define_mysql_order(self):
        
        self.sql_order = '''
        SELECT WIND_CODE, ANN_DT, REPORT_PERIOD, STATEMENT_TYPE, ACTUAL_ANN_DT, 
        TOT_OPER_REV, OPER_REV, NET_PROFIT_INCL_MIN_INT_INC, 
        NET_PROFIT_EXCL_MIN_INT_INC, NET_PROFIT_AFTER_DED_NR_LP, OPDATE 
        
        FROM [dbo].[ASHAREINCOME]
        '''
    
class CashflowStatement(FinancialData):
    unique_index = ['WIND_CODE', 'ACTUAL_ANN_DT', 'REPORT_PERIOD',
                    'STATEMENT_TYPE', 'OPDATE']
    name = 'CASHFLOWSTATEMENT'
    

    
    
    def define_mysql_order(self):
        
        self.sql_order = '''
        
        SELECT WIND_CODE, ANN_DT, REPORT_PERIOD, STATEMENT_TYPE,
        NET_CASH_FLOWS_OPER_ACT, FREE_CASH_FLOW, ACTUAL_ANN_DT, NET_INCR_CASH_CASH_EQU,
        OPDATE
        
        FROM [dbo].[ASHARECASHFLOW]
        '''
    


class IndexEOD(Data):
    lst = ['000001.SH', '000300.SH', '000905.SH', '000852.SH', '000016.SH']    
    unique_index = ['S_INFO_WINDCODE', 'TRADE_DT']
    name = 'INDEXEOD'
    

    
    @property    
    def getCalendarColumn(self):
        return 'TRADE_DT'
    
    @property    
    def getUpdateCalendarColumn(self):
        return 'TRADE_DT'

    
    def define_mysql_order(self):
        self.sql_order = '''
        SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_PRECLOSE, S_DQ_OPEN, S_DQ_HIGH, 
        S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME, S_DQ_AMOUNT
        FROM [dbo].[AINDEXEODPRICES]
        '''
    def update_mysql_order(self, begin_time):
        temp_lst = ['(S_INFO_WINDCODE = N\'{i}\'  )'.format(i = i) for i in self.lst]
        temp = 'OR'.join(temp_lst)
        temp = '(' + temp + ')'
        
        t = ' ({calendar_column} >= N\'{f}\') '.format(
            f = begin_time, calendar_column = self.getUpdateCalendarColumn)
        t = t + 'AND' + temp
        self.sql_order = self.sql_order + 'WHERE ' + t
        
        
    def download_mysql_order(self, begin_time):
        temp_lst = ['(S_INFO_WINDCODE = N\'{i}\'  )'.format(i = i) for i in self.lst]
        temp = 'OR'.join(temp_lst)
        temp = '(' + temp + ')'
        
        if self.getCalendarColumn != '':
            
            t = ' ({calendar_column} >= N\'{f}\') '.format(
                f = begin_time, calendar_column = self.getCalendarColumn)
            t = t + 'AND' + temp
            self.sql_order = self.sql_order + 'WHERE ' + t
            
        else:
            self.sql_order = self.sql_order + 'WHERE' + temp





class MarketCap(Data):
    unique_index = ['WIND_CODE', 'ANN_DT', 'CHANGE_DT', 'CHANGE_DT1', 'OPDATE']
    name = 'MARKETCAP'


    
    @property    
    def getCalendarColumn(self):
        return 'CHANGE_DT1'
    
    @property    
    def getUpdateCalendarColumn(self):
        return 'OPDATE'
    
    def define_mysql_order(self):

        self.sql_order = '''
        SELECT WIND_CODE, CHANGE_DT, TOT_SHR, FLOAT_SHR, S_SHARE_NTRD_PRFSHARE,
        ANN_DT, CHANGE_DT1, IS_VALID, OPDATE
        FROM [dbo].[ASHARECAPITALIZATION]
        '''    
    
    
class CDRMarketCap(Data):
    unique_index = ['WIND_CODE', 'ANN_DT', 'CHANGE_DT', 'CHANGE_DT1', 'OPDATE']
    name = 'CDRMARKETCAP'


    
    @property    
    def getCalendarColumn(self):
        return 'CHANGE_DT1'
    
    @property    
    def getUpdateCalendarColumn(self):
        return 'OPDATE'
    
    def define_mysql_order(self):

        self.sql_order = '''
        SELECT S_INFO_WINDCODE, CHANGE_DT, TOT_SHR, FLOAT_SHR, S_SHARE_NTRD_PRFSHARE,
        ANN_DT, CHANGE_DT1, IS_VALID, OPDATE
        FROM [dbo].[CDRCapitalization]
        '''       
        
    def process(self, df):
        df.columns = ['WIND_CODE'] + list(df.columns[1:])
        return df
        
    
    
    
        
class ST(Data):
    unique_index = ['S_INFO_WINDCODE', 'S_TYPE_ST', 'ENTRY_DT']
    name = 'ST'
    
    @property
    def getCalendarColumn(self):
        return 'ANN_DT'
    
    @property    
    def getUpdateCalendarColumn(self):
        return 'ANN_DT'
    
    def define_mysql_order(self):
    
        self.sql_order = '''
        SELECT S_INFO_WINDCODE, S_TYPE_ST, ENTRY_DT, REMOVE_DT, 
        ANN_DT, REASON
        FROM [dbo].[ASHAREST]
        '''    


class SWINDUSTRYCOMPONENT_OLD(Data):
    unique_index = ['S_INFO_WINDCODE', 'SW_IND_CODE', 'ENTRY_DT']
    name = 'SWINDUSTRYCOMPONENT'

    
    @property
    def getCalendarColumn(self):
        return 'ENTRY_DT'
    
    @property    
    def getUpdateCalendarColumn(self):
        return 'ENTRY_DT'
    
    def define_mysql_order(self):
    
        self.sql_order = '''
        SELECT S_INFO_WINDCODE, SW_IND_CODE, ENTRY_DT, REMOVE_DT
        FROM [dbo].[AShareSWIndustriesClass]
         '''    
class SWINDUSTRYCOMPONENT_NEW(Data):
    unique_index = ['S_INFO_WINDCODE', 'SW_IND_CODE', 'ENTRY_DT']
    name = 'SWINDUSTRYCOMPONENT_NEW'

    
    @property
    def getCalendarColumn(self):
        return 'ENTRY_DT'
    
    @property    
    def getUpdateCalendarColumn(self):
        return 'ENTRY_DT'
    
    def define_mysql_order(self):
    
        self.sql_order = '''
        SELECT S_INFO_WINDCODE, SW_IND_CODE, ENTRY_DT, REMOVE_DT
        FROM [dbo].[AShareSWNIndustriesClass]
        '''
        
        
        
class IndustryDict(Data):
    
    unique_index = ['INDUSTRIESCODE', 'INDUSTRIESNAME']
    name = 'INDUSTRYDICTIONARY'
    
    @property
    def getCalendarColumn(self):
        return 'OPDATE'
    
    @property    
    def getUpdateCalendarColumn(self):
        return 'OPDATE'
    
    def define_mysql_order(self):
    
        self.sql_order = '''
        SELECT INDUSTRIESCODE, INDUSTRIESNAME, LEVELNUM, USED, INDUSTRIESALIAS,
        WIND_NAME_ENG, INDUSTRIESCODE_OLD
        FROM [dbo].[AShareIndustriesCode]
        '''
        
    
    
    
# class Suspend(Data):
    
#     @property    
#     def getCalendarColumn(self):
#         return 'S_DQ_SUSPENDDATE'
    
#     @property    
#     def getUpdateCalendarColumn(self):
#         return 'OPDATE'
    
#     def define_mysql_order(self):

#         self.sql_order = '''
#         SELECT S_INFO_WINDCODE, S_DQ_SUSPENDDATE, S_DQ_SUSPENDTYPE, S_DQ_RESUMPDATE, OPDATE
#         FROM [dbo].[ASHARETRADINGSUSPENSION]
#         '''    






        
        
        
        

        
    


        
        
    
        
        
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        

        
    
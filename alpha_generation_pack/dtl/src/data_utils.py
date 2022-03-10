# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:34:55 2022

@author: Administrator
"""
import numpy as np
import datetime
from functools import wraps
import pandas as pd 

class DataProcess(object):
    
    def __init__(self):
        pass
    
    @staticmethod
    def time_logger(func, log_info = ''):
        @wraps(func)
        def wrapper(self, *arg, **kw):
            # print(func)
            if log_info == '':
                logged_info = str(func).split(' ')[1]
            else:
                logged_info = log_info
            if hasattr(self, 'verbose'):
                if self.verbose == 1:
                    print('----{log_info}----'.format(log_info = logged_info))
            
            n = datetime.datetime.now()
            res = func(self, *arg, **kw)
            
            if hasattr(self, 'verbose'):
                if self.verbose == 1:
                    print('{log_info}  used_time: {t}s'.format(
                        log_info = logged_info,
                        t = round((datetime.datetime.now() - n).total_seconds() ,2)
                        ))
                
                
            return res
        return wrapper

    
class SlidingWindowObject(object):
    
    def __init__(self,
                 code_field,
                 time_field,
                 period_field,
                 active_codes, 
                 active_dates, 
                 active_report_periods,
                 value_arr,
                 nearest_report_period,
                 start_report_period
                 ):
        
        self.code_field = code_field 
        self.time_field = time_field
        self.period_field = period_field
        self.active_codes = active_codes
        self.active_dates = active_dates
        self.active_report_periods = active_report_periods
        self.value_arr = value_arr
        
        self.nearest_report_period = nearest_report_period
        self.start_report_period = start_report_period
    
    
    
    
    def getUsedPosWithMinPeriods(self, min_periods):
        nearest_period_masked_pos = FinancialDataUtils.mapPos(
            self.nearest_report_period,
            [self.code_field, self.time_field, self.period_field],
            [self.active_codes, self.active_dates, self.active_report_periods],
            return_tuple = False)
        
    
        start_period_masked_pos = FinancialDataUtils.mapPos(
            self.start_report_period,
            [self.period_field],
            [self.active_report_periods],
            return_tuple = False)
        
            
        used_min_periods_boundary = min_periods - 1
        used_start_periods = start_period_masked_pos.reshape(-1) + \
            used_min_periods_boundary
        
        used_masked_pos = nearest_period_masked_pos[
            nearest_period_masked_pos[:,-1] >= used_start_periods]
        
        return used_min_periods_boundary, used_masked_pos
    
    def getFixedWindowArr(self, window):
                
        used_min_periods_boundary, used_masked_pos = self.getUsedPosWithMinPeriods(
            min_periods = window)
        
        fixed_window_arr = np.empty(shape = [used_masked_pos.shape[0], window])
        fixed_window_arr[:] = np.nan
        
        fixed_window_index_arr = np.empty(shape = [used_masked_pos.shape[0], 
                                                   used_masked_pos.shape[1] - 1])
        fixed_window_index_arr[:] = np.nan
        
        start = 0
        for i in range(used_min_periods_boundary, 
                       len(self.active_report_periods)):
            
            whole_slice, included_pos, excluded_pos = FinancialDataUtils.getSlidingWindowSlicedByLastColumn(
                used_masked_pos, i, window)
            
            sliced_values = self.value_arr[whole_slice]
            fixed_window_arr[start: start + len(sliced_values)] = sliced_values
            fixed_window_index_arr[start: start + len(sliced_values)] = included_pos[:, :-1]
            
            used_masked_pos = excluded_pos
            start += len(sliced_values)
        
        fixed_window_index_arr = fixed_window_index_arr.astype(int)
        codes = np.array(self.active_codes)[fixed_window_index_arr[:, 0]]
        dates = np.array(self.active_dates)[fixed_window_index_arr[:, 1]]
        
        index = pd.MultiIndex.from_arrays([dates, codes], names = ['日期','代码'])
        
        return index, fixed_window_arr
    
    

        
    
    


class FinancialDataUtils(DataProcess):
    
    '''
    没有考虑跳了一个report period的情况
    这个应该在forfill里面做到
    下个版本再考虑这个

    dropNone = Flase: 把 none 改成nan 再填0
            = True: 把 none 改成nan drop
    '''
    
    
    
    def __init__(self, orig_sheet,
                 setNan = True,
                 method = 'fill',
                 need_filter = True,
                 time_field = 'ACTUAL_ANN_DT',
                 code_field = 'WIND_CODE',
                 period_field = 'REPORT_PERIOD',
                 statement_field = 'STATEMENT_TYPE'):
        
        self.orig_sheet = orig_sheet
        self.time_field = time_field
        self.code_field = code_field
        self.period_field = period_field
        self.statement_field = statement_field
        self.setNan = setNan
        self.method = method
        # self.dropNone = dropNone
        # self.trade_time = trade_time
        self.verbose = 0
        self.need_filter = need_filter
        self.calibrateDates()
        if self.need_filter:
            self.filter_sheet()
    def reset(self):
        
        used_string_lst = [
            'sorted_sheet',
            'value_arr',
            'nearest_report_period',
            'start_report_period',
            'masked_arr'
            ]
        for attr in used_string_lst:
            self.__dict__.pop(attr, None)
        
        
    def filter_sheet(self):
        
        unique_codes = np.unique(self.orig_sheet[self.code_field])
        chosen_codes = filter(lambda x: x[0]!='A' and x.split('.')[1]!='BJ',
                              unique_codes)
        self.orig_sheet = self.orig_sheet[self.orig_sheet[self.code_field].isin(chosen_codes)]
        self.orig_sheet = self.orig_sheet[self.orig_sheet[self.period_field] > '20091201']
        
        season = ['0331', '0630', '0930', '1231']
        _min = str(self.orig_sheet[self.period_field].min())[:4]
        _max = str(self.orig_sheet[self.period_field].max())[:4]
        longlist = []
        for i in range(int(_min), int(_max) + 1):
            for j in season:
                longlist.append(str(i) + j)
        self.orig_sheet = self.orig_sheet[self.orig_sheet[self.period_field].isin(longlist)]
        
        
    def fetchCodesDatesRerportDates(self):
        
        self.active_codes = sorted(np.unique(
            self.orig_sheet[self.code_field].values))
        self.active_dates = sorted(np.unique(
            self.orig_sheet[self.time_field].values))
        self.active_report_periods = sorted(np.unique(
            self.orig_sheet[self.period_field].values))
    
    def calibrateDates(self):
        # time = pd.to_datetime(self.trade_time)
        self.orig_sheet.loc[:, self.time_field] = self.orig_sheet[self.time_field].replace([None], np.nan)
        self.orig_sheet = self.orig_sheet.dropna(subset = [self.time_field])
        self.orig_sheet.loc[:, self.time_field] = pd.to_datetime(self.orig_sheet[self.time_field].astype(str))
        # self.orig_sheet = self.orig_sheet[self.orig_sheet[self.time_field]<=time[-1] ]
        # self.orig_sheet.loc[:, self.time_field] = time[np.searchsorted(time, self.orig_sheet[self.time_field] )]
        

    
    def setSheetToConsolidatedStatement(self, chosen_field):
        
        res = self.orig_sheet[
            (self.orig_sheet['STATEMENT_TYPE'] == '408001000' ) |
            (self.orig_sheet['STATEMENT_TYPE'] == '408004000')]
        
        if self.setNan:
            if self.method == 'fill':
                res.loc[:, chosen_field] = res.loc[:, chosen_field].replace([None], 0)
            elif self.method == 'ignore':
                res.loc[:, chosen_field] = res.loc[:, chosen_field].replace([None], np.nan)
            elif self.method == 'drop':
                res = res.replace([None], np.nan).dropna(
                subset = [self.code_field, 
                self.time_field, 
                self.period_field,
                self.statement_field,
                chosen_field]
                )

        self.sorted_sheet = res.sort_values(by = [
            self.code_field, 
            self.time_field, 
            self.period_field,
            self.statement_field
            ])


    def setSheetToQuarterStatement(self, chosen_field):
        res = self.orig_sheet[
            (self.orig_sheet['STATEMENT_TYPE'] == '408002000' ) |
            (self.orig_sheet['STATEMENT_TYPE'] == '408003000')]
        
        
        if self.setNan:
            if self.method == 'fill':
                res.loc[:, chosen_field] = res.loc[:, chosen_field].replace([None], 0)
            elif self.method == 'ignore':
                res.loc[:, chosen_field] = res.loc[:, chosen_field].replace([None], np.nan)
            elif self.method == 'drop':
                res = res.replace([None], np.nan).dropna(
                subset = [self.code_field, 
                self.time_field, 
                self.period_field,
                self.statement_field,
                chosen_field]
                )


        self.sorted_sheet = res.sort_values(by = [
            self.code_field, 
            self.time_field, 
            self.period_field,
            self.statement_field
            ])
        
        
    
        
        
    
    def dropDuplicatedStatementsAtSameDate(self):
        #已经sorted 过 statement type, last取的是 4000/3000, 即调整
        #后报表

        temp  = self.sorted_sheet.set_index([
            self.code_field, 
            self.time_field, 
            self.period_field,
            ])
        temp = temp[~temp.index.duplicated(keep = 'last')]
        self.sorted_sheet = temp.reset_index()
        




    def createTemplate(self, fill = np.nan):
        t = np.empty(shape = [ 
            len(self.active_codes), 
            len(self.active_dates), 
            len(self.active_report_periods)
            ] )
        
        t[:] = fill
        return t
    
    @staticmethod
    def createPosition(arr):
        return dict(zip(arr, range(len(arr))))
    
    @staticmethod
    def mapPos(sheet, field_names, field_values, return_tuple = True):
        
        mapped_lst = [ sheet[name].map(
            FinancialDataUtils.createPosition(val)
            ).values.reshape(-1,1)
            for name, val in zip(field_names, field_values)]
        
        
    
        mapped_pos = np.concatenate(mapped_lst, axis = -1)
        
        if return_tuple:
            return tuple([mapped_pos[:,i]  for i in range(mapped_pos.shape[-1]) ] )
        else:
            return mapped_pos
        
    def setSheetValue(self, chosen_field):
        
        self.value_arr = self.createTemplate()

        self.value_arr[
            FinancialDataUtils.mapPos(
                self.sorted_sheet,
                [self.code_field, self.time_field, self.period_field],
                [self.active_codes, self.active_dates, self.active_report_periods]
                )
                   ] = self.sorted_sheet[chosen_field].values

    @staticmethod
    def checkfirstQuarter(report_period):
        time = pd.to_datetime(str(report_period)).month
        
        return time== 3
        
    
    def findFirstQuarterPosition(self):
        lst = []
        for i in range(len(self.active_report_periods)):
            temp_period = self.active_report_periods[i]
            lst.append(FinancialDataUtils.checkfirstQuarter(temp_period))
        
        return lst
    
    
    def setConsolidatedStatementToSingleQuarter(self):
        
        temp = np.diff(self.value_arr, axis = -1, prepend = np.nan)
        first_quarter_pos = self.findFirstQuarterPosition()
        temp[:, :, first_quarter_pos] = self.value_arr[:, :, first_quarter_pos]
        
        self.value_arr = temp
        del temp
    
    
    def setSingleQuarterStatementToConsolidated(self):
    
        temp = self.value_arr.copy()
        first_quarter_pos = self.findFirstQuarterPosition()
        for i in range(len(first_quarter_pos)):
            if not first_quarter_pos[i] and i > 0:
                temp[:, :, i] = temp[:, :, i] + temp[:, :, i-1]
        self.value_arr = temp
        del temp
    
        
    
    
    @staticmethod
    def forwardfillNA(sheet, axis = 1, replace = True):
        if not replace:
            sheet = sheet.copy()
        
        #will replace the orignal sheet 
        for i in range(1, sheet.shape[axis]):            
            
            sl = [slice(None)] * sheet.ndim
            sl[axis] = i
            
            prev_sl = [slice(None)] * sheet.ndim
            prev_sl[axis] = i -1
            
            sl, prev_sl = tuple(sl), tuple(prev_sl)
            nan_pos = np.isnan( sheet[sl] )
            sheet[sl][nan_pos] = sheet[prev_sl][nan_pos]
        
        return sheet


    def setNearestReportPeriod(self):
        #因为会有同一天有两个财报的信息所以取最大的那个
        #因为已经cummax排好序了，所以直接取last就行了
        self.nearest_report_period = self.sorted_sheet[[
                                            self.code_field, 
                                            self.time_field, 
                                            self.period_field]].set_index(
                                                [self.code_field, self.time_field]
                                                ).groupby(
                                                [self.code_field]
                                                )[self.period_field].apply(
                                                    lambda x: x.cummax())
                                                    
        self.nearest_report_period = self.nearest_report_period[
            ~self.nearest_report_period.index.duplicated(keep = 'last')].reset_index()    
    
    def setStartReportPeriod(self):
        
        
        self.start_report_period = self.sorted_sheet[[
                                            self.code_field, 
                                            self.time_field, 
                                            self.period_field]].set_index(
                                                [self.code_field, self.time_field]
                                                ).groupby(
                                                [self.code_field]
                                                )[self.period_field].apply(
                                                    lambda x: x.cummin())
    
        self.start_report_period = self.start_report_period[
            ~self.start_report_period.index.duplicated(keep = 'first')].reset_index()
            

    def getUsedPosWithMinPeriods(self, min_periods):
        nearest_period_masked_pos = FinancialDataUtils.mapPos(
            self.nearest_report_period,
            [self.code_field, self.time_field, self.period_field],
            [self.active_codes, self.active_dates, self.active_report_periods],
            return_tuple = False)
        
    
        start_period_masked_pos = FinancialDataUtils.mapPos(
            self.start_report_period,
            [self.period_field],
            [self.active_report_periods],
            return_tuple = False)
        
            
        used_min_periods_boundary = min_periods - 1
        used_start_periods = start_period_masked_pos.reshape(-1) + \
            used_min_periods_boundary
        
        used_masked_pos = nearest_period_masked_pos[
            nearest_period_masked_pos[:,-1] >= used_start_periods]
        
        return used_min_periods_boundary, used_masked_pos
    
    
    
    
    
    @staticmethod
    def getSlidingWindowSlicedByLastColumn(used_masked_pos, period_pos, window):
        
        choosed_index = used_masked_pos[:,-1] == period_pos
        end_sliced_pos = period_pos + 1
        start_sliced_pos = max(0, period_pos + 1 - window)
        slice_for_this_period = slice(start_sliced_pos, end_sliced_pos)

        sliced_position = used_masked_pos[:,:-1][choosed_index]
        whole_slice = list(sliced_position.T)
        
        whole_slice.append(slice_for_this_period)
        whole_slice = tuple(whole_slice)
        
        return  whole_slice, \
                used_masked_pos[choosed_index], \
                used_masked_pos[~choosed_index]        
    
    
    
    def setSheetMasked(self, window, min_periods = 0):
    
        if min_periods <= 0:
            min_periods = window
        
        self.masked_arr = self.createTemplate()
        used_min_periods_boundary, used_masked_pos = self.getUsedPosWithMinPeriods(
            min_periods)
        
        
        
        for i in range(used_min_periods_boundary, 
                       len(self.active_report_periods)):
            
            whole_slice, _, excluded_pos = FinancialDataUtils.getSlidingWindowSlicedByLastColumn(
                used_masked_pos, i, window)            
            self.masked_arr[whole_slice] = False
            used_masked_pos = excluded_pos
        
        self.masked_arr[np.isnan(self.masked_arr)] = True
    
    
    
    
    def getFixedWindowArr(self, window):
                
        used_min_periods_boundary, used_masked_pos = self.getUsedPosWithMinPeriods(
            min_periods = window)
        
        fixed_window_arr = np.empty(shape = [used_masked_pos.shape[0], window])
        fixed_window_arr[:] = np.nan
        
        fixed_window_index_arr = np.empty(shape = [used_masked_pos.shape[0], 
                                                   used_masked_pos.shape[1] - 1])
        fixed_window_index_arr[:] = np.nan
        
        start = 0
        for i in range(used_min_periods_boundary, 
                       len(self.active_report_periods)):
            
            whole_slice, included_pos, excluded_pos = FinancialDataUtils.getSlidingWindowSlicedByLastColumn(
                used_masked_pos, i, window)
            
            sliced_values = self.value_arr[whole_slice]
            fixed_window_arr[start: start + len(sliced_values)] = sliced_values
            fixed_window_index_arr[start: start + len(sliced_values)] = included_pos[:, :-1]
            
            used_masked_pos = excluded_pos
            start += len(sliced_values)
        
        fixed_window_index_arr = fixed_window_index_arr.astype(int)
        codes = np.array(self.active_codes)[fixed_window_index_arr[:, 0]]
        dates = np.array(self.active_dates)[fixed_window_index_arr[:, 1]]
        
        index = pd.MultiIndex.from_arrays([codes, dates], names = ['代码','日期'])
        
        return index, fixed_window_arr
    
    
    @DataProcess.time_logger
    def intializeConsolidatedStatement(self, chosen_field):
        self.reset()
        self.setSheetToConsolidatedStatement(chosen_field)
        self.dropDuplicatedStatementsAtSameDate()
        
    @DataProcess.time_logger
    def intializeQuarterStatement(self, chosen_field):
        self.reset()
        self.setSheetToQuarterStatement(chosen_field)
        self.dropDuplicatedStatementsAtSameDate()    
    

    @DataProcess.time_logger
    def setValue(self, chosen_field):
        self.fetchCodesDatesRerportDates()
        self.setSheetValue(chosen_field)        
        FinancialDataUtils.forwardfillNA(self.value_arr, axis = 1, replace = True)
    
    
    
    @DataProcess.time_logger
    def setMasked(self, *args, **kargs):
        self.setNearestReportPeriod()
        self.setStartReportPeriod()
        self.setSheetMasked(*args, **kargs)

    @DataProcess.time_logger
    def slidingWindow(self, *args, **kargs):
        self.setNearestReportPeriod()
        self.setStartReportPeriod()
        
        return self.getFixedWindowArr(*args, **kargs)       

        
    @DataProcess.time_logger
    def slidingWindowConsolidatedStatement(self, 
                                           chosen_field,
                                           *args, 
                                           to_single_season = False,
                                           **kargs):
        
        self.intializeConsolidatedStatement(chosen_field)
        self.setValue(chosen_field)
        if to_single_season:
            self.setConsolidatedStatementToSingleQuarter()
        self.setMasked(*args, **kargs)        
        
        return np.ma.masked_array(self.value_arr, 
                     mask = self.masked_arr)
        
    
    @DataProcess.time_logger
    def fixedSlidingWindowConsolidatedStatement(self, 
                                                chosen_field, 
                                                *args, 
                                                single_quarter = False,
                                                **kargs):
        
        self.intializeConsolidatedStatement(chosen_field)
        self.setValue(chosen_field)
        if single_quarter:
            self.setConsolidatedStatementToSingleQuarter()
            
        return self.slidingWindow(*args, **kargs)
        
    @DataProcess.time_logger
    def rolling(self, 
                      chosen_field, 
                      *args, 
                      single_quarter = True,
                      return_obj = False,
                      use_single_quarter_info = True,
                      **kargs):
        
        if not use_single_quarter_info:
            return self.fixedSlidingWindowConsolidatedStatement(
                chosen_field, *args, single_quarter = single_quarter, **kargs
                )
        
        self.intializeQuarterStatement(chosen_field)
        self.setValue(chosen_field)
        temp_saved_arr = self.value_arr.copy()
        self.intializeConsolidatedStatement(chosen_field)
        self.setValue(chosen_field)
        self.setConsolidatedStatementToSingleQuarter()
        fill_pos = np.logical_and(
            np.isnan(self.value_arr),
            ~np.isnan(temp_saved_arr))
        self.value_arr[fill_pos] = temp_saved_arr[fill_pos]
        if not single_quarter:
            self.setSingleQuarterStatementToConsolidated()
        
        
        if not return_obj:
            return self.slidingWindow(*args, **kargs)
        else:
            self.setNearestReportPeriod()
            self.setStartReportPeriod()
            
            return SlidingWindowObject(
                self.code_field,
                self.time_field,
                self.period_field,
                self.active_codes, 
                self.active_dates, 
                self.active_report_periods,
                self.value_arr,
                self.nearest_report_period,
                self.start_report_period
                )
        
    
    def nearest(self, chosen_field):
        self.intializeConsolidatedStatement(chosen_field)
        value_df = self.sorted_sheet[[
                                        self.code_field, 
                                        self.time_field, 
                                        self.period_field,
                                        chosen_field]].set_index(
                                            [self.code_field, self.time_field]
                                            )
        nearest_period_df = value_df.groupby(
                                [self.code_field]
                                )[self.period_field].apply(
                                    lambda x: x.cummax())
        
        nearest_pos = value_df[self.period_field].values == nearest_period_df.values
        
        #REPORT_PERIOD也 sort过，所以取后一个就行了
        final_res = value_df[nearest_pos].reset_index().set_index([
            self.time_field, self.code_field])            
        final_res = final_res[~final_res.index.duplicated(keep = 'last')]        
        
        return final_res[chosen_field]
        


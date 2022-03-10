# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:23:55 2022

@author: Administrator
"""

import datetime
import pandas as pd
from tqdm import tqdm




class Data(object):
    def __init__(self, conn):
        self.conn = conn
    
    def get_trading_datelist(self, begT, endT):
        begT = int(begT)
        endT = int(endT)
        code = '''
        function get_trade(begt,endt);
        begin
           return 
           select * from infotable 753 of 'SH000001' where ['截止日'] >= begt and ['截止日'] <= endt end;
        end;
        '''
        r = self.conn.conn.call('get_trade', begT, endT, code=code)
        datelist = pd.DataFrame(r.value())
        datelist = datelist['截止日'].tolist()
        return datelist


class IndexWeight(Data):
    lst = ['000300.SH', '000905.SH', '000852.SH', '000016.SH']    
    
    unique_index = ['截止日','指数代码', '代码']
    name = 'INDEXWEIGHT'    
    
    @property
    def getUpdateCalendarColumn(self):
        return '截止日'


    def fetchone(self, index, date):
        date = int(date)
        code = '''
        function get_components(index_code,endt);
        begin
            getbkweightbydate(index_code,inttodate(endt),t); 
            Return t;
        end;
        '''
        r = self.conn.conn.call('get_components', index, date, code=code)
        return r.dataframe()

    
    
    def download(self, begin_time):
        
        end_time = str(pd.to_datetime(datetime.datetime.now()))[:10]
        end_time = ''.join(end_time.split('-'))
        date_lst = self.get_trading_datelist(begin_time, end_time)
        
        lst = []
        for i in tqdm(date_lst):
            for j in self.lst:
                j = j.split('.')[1] + j.split('.')[0]
                try:
                    one = self.fetchone(j, i)[['截止日','代码', '指数代码','比例(%)']]
                except:
                    continue
                lst.append(one)
            
        return pd.concat(lst)

    def update(self, begin_time):
        
        end_time = str(pd.to_datetime(datetime.datetime.now()))[:10]
        end_time = ''.join(end_time.split('-'))
        date_lst = self.get_trading_datelist(begin_time, end_time)
        
        lst = []
        '''
        不包括
        '''
        for i in date_lst[1:]:
            for j in self.lst:
                j = j.split('.')[1] + j.split('.')[0]
                one = self.fetchone(j, i)[['截止日','代码', '指数代码','比例(%)']]
                lst.append(one)
          
        if len(lst) == 0:
            return pd.DataFrame([])
        else:          
            return pd.concat(lst)



class QuarterMinBar(Data):
    
    unique_index = ['date,StockID']
    name = 'QUARTERMINBAR'    
    

        
    def getstocklist(self):
        
        return [i.split('.')[1] + i.split('.')[0] for i in self.stock_list.values]
    
    
    
    @property
    def getUpdateCalendarColumn(self):
        return 'date'


    # def fetchone(self, stock, date):
    #     one =  self.conn.conn.query(stock = stock, 
    #                    cycle = '15分钟线', 
    #                    begin_time = date, 
    #                    end_time = date,
    #                    fields = ['date,StockID,open,close,high,low,vol,amount,cjbs,yclose'])
    #     return one

    def download(self, begin_time, end_time):

        begin_time =  int(''.join(str(pd.to_datetime(begin_time))[:10].split('-')))
        end_time =  int(''.join(str(pd.to_datetime(end_time))[:10].split('-')))

        res = self.conn.conn.query(stock  = self.getstocklist(),
                             cycle = '15分钟线',
                             begin_time = begin_time,
                             end_time = end_time,
                             fields = 'date,StockID,open,close,high,low,vol,amount,cjbs,yclose')
        
        # lst = []
        # for i in tqdm(date_lst):
        #     j = self.get_stocklist(i)
        #     try:
        #         one = self.fetchone(j, int(i)) 
        #     except:
        #         continue
        #     lst.append(one)
            
        return res.dataframe()

        
        
    def update(self, begin_time):
        begin_time =  int(''.join(str(pd.to_datetime(begin_time))[:10].split('-')))
        end_time = str(pd.to_datetime(datetime.datetime.now() +datetime.timedelta(days = 1)  ))[:10]
        end_time = ''.join(end_time.split('-'))
        end_time = int(end_time)        
        
        res = self.conn.conn.query(stock  = self.getstocklist(),
                             cycle = '15分钟线',
                             begin_time = begin_time,
                             end_time = end_time,
                             fields = 'date,StockID,open,close,high,low,vol,amount,cjbs,yclose')
        return res.dataframe()
                
        
        
        
    
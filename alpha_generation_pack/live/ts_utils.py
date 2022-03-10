import numpy as np
import pandas as pd 
import os
import datetime


def getIOVolume(w, prev, now):
    df = pd.concat([prev, now], axis = 1).fillna(0)
    cross = df.iloc[:,1] - df.iloc[:,0]
    buy = cross[cross > 0]
    sell = cross[cross <0]
    return getTradePlatFormat(w, buy), getTradePlatFormat(w, sell)


def getTradePlatFormat(w, df):
    if len(df) == 0:
        return None
    date = str(pd.to_datetime(datetime.datetime.now()))[:10]
    error, res = w.wsd(','.join(df.index), 'sec_name', date, date, usedf = True)
    df = pd.DataFrame(df)
    df.loc[:, '证券简称'] = res
    df = df.reset_index().set_index(['index','证券简称']).iloc[:,0]
    
    
    lst = []
    market_lst  = []
    df.index.names = ['代码','证券简称']
    df.name = '证券数量'
    df = df.reset_index()
    for i in df['代码']:
        lst.append(i.split('.')[0])
        market = i.split('.')[1]
        if market == 'SH':
            market_lst.append('上海A股')
        else:
            market_lst.append('深圳A股')
    df = pd.DataFrame(df)
    # 市场代码	证券代码	证券简称	证券类型	买卖方向

    
    df['市场代码'] = market_lst
    df['证券代码'] = lst

    df['证券类型'] = 1
    df['买卖方向'] = 1
    for i in ['溢价比例', '折价比例', '每股股利（当月）', '每股股利（下月）', '每股股利（第一个季月）', '每股股利（第二个季月）']:
        df[i] = np.nan

    
    df = df [['市场代码', '证券代码','证券简称','证券类型', '买卖方向','证券数量',	'溢价比例', '折价比例', '每股股利（当月）', '每股股利（下月）', '每股股利（第一个季月）', '每股股利（第二个季月）']]
    return df.set_index('市场代码')


def runOrLoad(root, name, func):
    if not os.path.exists(root):
        os.makedirs(root)
    
    file = os.path.join(root, name + '.csv')
    if os.path.exists(file):
        return pd.read_csv(file,index_col = 0).T.iloc[:, 0]
    else:
        f = func()
        f.to_csv(file)
        return f.T.iloc[:, 0]


def checkIndex(df):
    lst = []
    for i in df.index:
        if len(i.split('.')) == 1:
            i = i[2:] + '.' + i[:2]
            
        if i.split('.')[1] == 'NE':
            lst.append(i.split('.')[0] + '.' + 'BJ')
        elif i.split('.')[1] == 'SS':
            lst.append(i.split('.')[0] + '.' + 'SH')
        else:
            lst.append(i)
    
    df.index = lst
    return df

def concatPrice(prev_price, now_price):
    
    res = prev_price.copy()
    now_price = checkIndex(now_price)
    
    res.loc[set(now_price.index) - set(prev_price.index)] = np.nan
    res.loc[now_price.index] = now_price
    return res



def concatSuspend(suspendYS, suspendTD):
    
    res = suspendYS.copy()
    
    suspendTD = checkIndex(suspendTD)
    suspendTD = suspendTD.dropna().astype(bool)
    res.loc[set(suspendTD.index) - set(suspendYS.index)] = np.nan
    naindex = res[res.isna()].index
    res.loc[suspendTD[suspendTD == True].index] = True
    res.loc[suspendTD[suspendTD == False].index] = False
    res.loc[naindex] = np.nan
    return res

def getLiveSTAndSuspend(conn, stocks, date):
    date = int(''.join(str(pd.to_datetime(date))[:10].split('-')))
    r = conn.call('indicator', ';'.join(stocks), date, code= '''
    function indicator(stocklist,endt);
        
        begin
        SetSysParam("CurrentDate",inttodate(endt));
        SetSysParam("Cycle","日线");
        SetSysParam("bRate",0);
        SetSysParam("RateDay",0.0);
        SetSysParam("Precision",2);
        SetSysParam("profiler",0);
        SetSysParam("ReportMode",0);
        SetSysParam("EmptyMode",0);
        SetSysParam("CalcCTRLWord",0);
        SetSysParam("LanguageID",0);
        setsysparam(pn_date(),inttodate(endt)); 
        endt := inttodate(endt);
        return 
        
        Query("",stocklist,True,"","Code",DefaultStockID(),
        "IsSuspend",StockSNIsSuspend(endt),
        "IsST",IsST_3(endt)) ;
        end;
    ''')

    return checkIndex(r.dataframe().set_index('Code'))

def getLivePrice(conn, stocks, date):
    date = ''.join(str(pd.to_datetime(date))[:10].split('-')) + 'T'
    stocks_str = ';'.join(stocks)
    code = """
        r:=array(); 
        begt:={dt}; 
        endt:={dt}; 
        stocks:= '{st}';
        bt:=9/24+24/24/60; 
        et:=9.4999/24; 
        t:=MarketTradeDayQk(BegT,EndT); 

        for i:=0 to length(t)-1 do 
        begin 
             vEndt:=t[i]; 
             r&=select ['StockID'],['price'] 
             from tradetable datekey vEndt+bt to vEndt+et of stocks
                 where ['vol']>0 end; 
        end 
        return r; 
    """.format(dt = date, 
                st = stocks_str)
    r = conn.exec(code)
    return checkIndex(r.dataframe().set_index('StockID')['price'])
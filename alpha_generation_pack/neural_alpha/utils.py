from itertools import chain
from functools import reduce
from neural_alpha.data_generation import StockMinutePanel, StockMinuteSeries
import pandas as pd 
from tqdm import tqdm
import numpy as np 

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



def addFeatures(df, dataloader):
    check_num = 16
    
    df = df.copy()
    df['datet'] = pd.to_datetime(df.index.date)
    df['StockID'] = [(i[2:] + '.' + i[:2]).replace('NE', 'BJ') for i in df['StockID']]
    df['vwap'] = df['amount']/df['vol']
    df['vwap'] = df['vwap'].fillna(df['close'])
    
    df = df.set_index(['StockID', 'datet'])
    
    adj = dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_ADJFACTOR')
    shares = dataloader.read('MIXED', 'calculated', 'CAPS', 'FLOAT_SHR').astype(float)

    dup_ind = ~df.index.duplicated(keep = 'first')
    if (dup_ind).sum() * check_num == len(df):
        adj = np.broadcast_to(np.expand_dims(adj.unstack().loc[df.index[dup_ind]].values, 
                                             axis = 1), 
                              ((dup_ind).sum() , check_num)).reshape(-1, 1)
        
        shares = np.broadcast_to(np.expand_dims(shares.unstack().loc[df.index[dup_ind]].values, 
                                             axis = 1), 
                              ((dup_ind).sum() , check_num)).reshape(-1, 1)
        df['adj'] = adj
        df['shares'] = shares
    else:
        adj = adj.loc[df.index.levels[1]].unstack()
        adj.name = 'adj'
        shares = shares.loc[df.index.levels[1]].unstack()
        shares.name = 'shares'
        
        t = pd.concat([adj, shares], axis = 1)
        chosen = t.index.get_indexer(df.index)
        df[t.columns] = t.values[chosen, :]
        if len(chosen[chosen==-1])!=0:
            df.iloc[np.where(chosen == -1),:] = np.nan
    df = df.reset_index().set_index(['StockID', 'datet', 'date'])
    return df


def getLabel(dataloader, benchmark,
             excess = True, predict_len = 5):
    
    adj = dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_ADJFACTOR')
    close = dataloader.read('WINDDB', 'calculated', 'PVProcess', 'S_DQ_CLOSE')
    stock_ret = (adj*close).pct_change(periods = predict_len).shift(-predict_len)
    if not excess:
        return stock_ret
    
    market_close = dataloader.getMarketInfo(field = 'S_DQ_CLOSE', 
                      market_index = benchmark)
    future_benchmark_return = market_close.astype(float).pct_change(
        periods = predict_len).shift(-predict_len)    
    
    return (stock_ret.T - future_benchmark_return).T   

def getMasked(dataloader):
    return dataloader.getSuspend().fillna(1).astype(int)
    
    


def getStockPanel(df, label, masked_valid_series,
                  label_name = ['label_0'], test = True):
    
    lst = []
    for i in tqdm(df.index.levels[0]):
        this_data = df.loc[i]
        if not test:
            this_label = label.loc[this_data.index.levels[0], i]
        else:
            this_label = pd.Series(0, index = this_data.index.levels[0] )
        
        this_label.name = label_name[0]
        this_masked = masked_valid_series.loc[this_data.index.levels[0], i]
        this_masked.name = 'masked'
        
        if not len(this_data)/len(this_data.index.remove_unused_levels().levels[0]) == StockMinuteSeries.check_num:
            t = this_data.groupby(level = 0)['open'].count()
            this_data = this_data.loc[t[t==StockMinuteSeries.check_num].index]
        
        lst.append(StockMinuteSeries(this_data, 
                                     pd.DataFrame(this_label),
                                     pd.DataFrame(this_masked),
                                     i))
    
    return StockMinutePanel(lst)




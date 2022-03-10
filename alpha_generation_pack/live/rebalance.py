import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import portfolio.optimizer
import dtl.src.loader
import factors.modelbased_factor
import pandas as pd
import dtl.src.toolbox
import dtl.src.connection
import live.ts_utils
from WindPy import w as wpy
import datetime




now_date = input('Date Rebalance, press Enter if it is today')
prev_date = input('Date Rebalance Previous, press Enter if it is last tradingday')
is_first = False


wpy.start()
if now_date == '':
    now_date = str(pd.to_datetime(datetime.datetime.now()))[:10]


dataloader = dtl.src.loader.DataLoader()
barraloader = dtl.src.loader.BarraLoader( r'Z:\LuTianhao\Barra' )

if prev_date == '':
    try:
        prev_date = list(dataloader.getTradeDays()).index(pd.to_datetime(now_date))
    except:
        prev_date = str(list(dataloader.getTradeDays())[-1])[:10]

temp_root = r'H:\alpha_generation_pack\live\temp'
opt = portfolio.optimizer.OnlineBarraOptimizer(dataloader, barraloader, prev_date)



alphaforecast = live.ts_utils.runOrLoad(temp_root, 'alpha', 
                                        lambda : factors.modelbased_factor.AlphaNetFactor(
    r'H:/alpha_generation_pack/neural_alpha/models/alphenetV3.5/alphanetV3_181224.h5', 
                                             dataloader).predict_one(prev_date))

universe = live.ts_utils.runOrLoad(temp_root, 'universe',
                                   lambda : dtl.src.toolbox.StockUniverse.getAvailableStockUniverse(
                                       dataloader).loc[[prev_date]])
if is_first:
    prev_volume = pd.Series([])
else:
    prev_volume = live.ts_utils.runOrLoad(temp_root, 'prev_volume',
                                          lambda : pd.read_csv(
                                              os.path.join('Z:\LuTianhao\日频策略跟踪\组合权重',
                                                           prev_date + '.csv'
                                                           ),index_col = 0).T)
    prev_volume = prev_volume[prev_volume!=0]
prev_suspend = live.ts_utils.runOrLoad(temp_root, 'prev_suspend',
                                      lambda : dataloader.getSuspend().loc[[prev_date]])

prev_close = live.ts_utils.runOrLoad(temp_root, 'prev_close',
                                      lambda : dataloader.read('WINDDB',
                                                               'calculated',
                                                               'PVProcess',
                                                               'S_DQ_CLOSE').loc[[prev_date]])

#%%

conn = dtl.src.connection.TinySoftData()
non_available_today = live.ts_utils.getLiveSTAndSuspend(conn.conn, universe.index, now_date)
price_today = live.ts_utils.getLivePrice(conn.conn, universe.index, now_date)
conn.conn.logout()

input_universe = universe.copy()
input_universe.loc[non_available_today.astype(bool).any(axis = 1)] = False


suspend = live.ts_utils.concatSuspend(prev_suspend, non_available_today['IsSuspend'])
forecast = alphaforecast.loc[set(input_universe[input_universe == True].index).intersection(set(alphaforecast.index))]

price = live.ts_utils.concatPrice(prev_close, price_today)
mv = (prev_volume * price.loc[prev_volume.index]).sum()
last_weight = (prev_volume * price.loc[prev_volume.index])/mv

if is_first:
    opt.iterate = 0
    mv = float(input('Please Enter MarketValue of the portfolio'))
w = opt.optimize(last_weight, forecast, suspend)
w = w[w!=0]
new_volume = mv * w/price.loc[w.index]
new_volume = new_volume//100 * 100
new_volume.name = now_date
new_volume.to_csv(os.path.join('Z:\LuTianhao\日频策略跟踪\优化清单\\' , now_date + '.csv'))

buylist, selllist = live.ts_utils.getIOVolume(wpy, prev_volume, new_volume)
if not buylist is None:
    buylist.to_excel(r'Z:\LuTianhao\日频策略跟踪\调整清单\{t}_买入清单.xlsx'.format(t = now_date))
if not selllist is None:    
    selllist.to_excel(r'Z:\LuTianhao\日频策略跟踪\调整清单\{t}_卖出清单.xlsx'.format(t = now_date))



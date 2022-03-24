
import peval.pmvEval
import utils.pandas_utils
import numpy as np 
from tqdm import tqdm


class PortfolioSortingUtils(object):
    
    
    @staticmethod
    def getGroupedPortfolio(factor, num):
        cut = [i/num for i in range(num + 1)]
        ranked_factor = factor.rank(pct = True,  axis = 1)
        df = ranked_factor.copy()
        for i in range(1, num + 1):
            df[(cut[i-1] < ranked_factor) & (ranked_factor<= cut[i])] = i
        
        return df
    
    
    @staticmethod
    def getPortfolioWeight(grouped_df, group_id):
        grouped_df = grouped_df == group_id
        return (grouped_df.T/grouped_df.T.sum()).T



def univariateSort(factor,
                   dataloader,
                   universe = 'NOBJ',
                   group_num = 5,
                   remove_st = True,
                   remove_suspend = True,
                   remove_zt = True,
                   remove_dt = True,
                   show_progress = True,
                   industry_netural = False):
    
    if remove_st:
        remove_st = dataloader.getST().fillna(1).astype(int)
    
    if remove_suspend:
        remove_suspend = dataloader.getSuspend().fillna(1).astype(int)
        
    if remove_zt:
        remove_zt = dataloader.getSPZT().fillna(1).astype(int)
        
    if remove_dt:
        remove_dt = dataloader.getSPDT().fillna(1).astype(int)
    
    
    if universe == 'NOBJ':
        choosed = []
        for i in factor.columns:
            if i.split('.')[1] != 'BJ':
                choosed.append(i)
        
        factor = factor.loc[:, choosed].sort_index(axis = 1)

    adjclose = dataloader.getAdjClose().fillna(0)
    
    
    if not industry_netural:
        group_df = PortfolioSortingUtils.getGroupedPortfolio(factor, group_num)
        lst = [peval.pmvEval.WeightPosition(PortfolioSortingUtils.getPortfolioWeight(group_df, i + 1)) 
               for i in range(group_num)]
        portfolios = peval.pmvEval.WeightPositions(*lst)
        z = peval.pmvEval.PortfolioResult(portfolios)
        res_mv, res_volume = z.getPortfolioMV(adjclose,
                         remove_st = remove_st,
                         remove_suspend = remove_suspend,
                         remove_zt = remove_zt,
                         remove_dt = remove_dt,
                         show_progress = show_progress)
        
        return res_mv ,res_volume

    else:
        industry = dataloader.getIndustry().unstack()
        ind_cls = industry.value_counts().index
        
        dic_mv = {}
        dic_volume = {}
        for ind in tqdm(ind_cls, 
                        desc = '行业分组回测中',
                        total = len(ind_cls),
                        disable = not show_progress):
            mytempind = industry[industry == ind].unstack().T
            mytempind, mytempfac = utils.pandas_utils._align(mytempind, factor)
            mytempind = mytempind.isna().astype(int).replace([1,0], [np.nan,1])
            mytempfac = mytempind * mytempfac
            
            
            group_df = PortfolioSortingUtils.getGroupedPortfolio(mytempfac, group_num)
            lst = [peval.pmvEval.WeightPosition(PortfolioSortingUtils.getPortfolioWeight(group_df, i + 1)) 
                   for i in range(group_num)]
            portfolios = peval.pmvEval.WeightPositions(*lst)
            z = peval.pmvEval.PortfolioResult(portfolios)
            res_mv, res_volume = z.getPortfolioMV(adjclose,
                             remove_st = remove_st,
                             remove_suspend = remove_suspend,
                             remove_zt = remove_zt,
                             remove_dt = remove_dt,
                             show_progress = False)
                
            dic_mv[ind] = res_mv
            dic_volume[ind] = res_volume
        return dic_mv, dic_volume


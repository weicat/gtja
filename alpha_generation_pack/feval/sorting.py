
import peval.pmvEval

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
                   remove_dt = True):
    
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
            if i.split('.')[1] == 'BJ':
                choosed.append(i)
        
        factor = factor[choosed].sort_index(axis = 1)

    
    
    adjclose = dataloader.getAdjClose().fillna(0)
    group_df = PortfolioSortingUtils.getGroupedPortfolio(factor, group_num)
    
    lst = [peval.pmvEval.WeightPosition(PortfolioSortingUtils.getPortfolioWeight(group_df, i + 1)) 
           for i in range(group_num)]
    portfolios = peval.pmvEval.WeightPositions(*lst)
    z = peval.pmvEval.PortfolioResult(portfolios)
    res_return, res_volume = z.getPortfolioMV(adjclose,
                     remove_st = remove_st,
                     remove_suspend = remove_suspend,
                     remove_zt = remove_zt,
                     remove_dt = remove_dt)
    
    return res_return ,res_volume
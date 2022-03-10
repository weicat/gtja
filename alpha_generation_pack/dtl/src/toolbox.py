# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:18:34 2022

@author: Administrator
"""
import dtl.src.new_utils as utils
from tqdm import tqdm
import pandas as pd 
import numpy as np
import cvxpy as cp
import statsmodels.api as sm
import joblib
from numba import njit

@njit
def countTrailingSuspendDaysOneStock(df):
    count = 0
    lst = np.zeros(len(df))
    for i in range(len(df)):
        if df[i] == 0:
            count = 0
            continue
        else:
            count +=1
            lst[i] = count
    
    return lst


class ToolBox(object):
    pass


class BarraToolBox(ToolBox):
    pass

class StockUniverse(ToolBox):
    
    
    @staticmethod
    def countTrailingSuspendDays(isSuspend):
        r = []
        isSuspend = isSuspend.fillna(method = 'ffill')
        isSuspend = isSuspend.astype(float).fillna(0)
        for i in range(isSuspend.shape[1]):
            r.append(countTrailingSuspendDaysOneStock(isSuspend.iloc[:,i].values))
        
        r = pd.DataFrame(r, index = isSuspend.columns, columns = isSuspend.index)
        return r.T

    @staticmethod
    def getLongSuspendTF(isSuspend, longSuspendDays, backDays):
        trailingSuspendDays = StockUniverse.countTrailingSuspendDays(isSuspend)
        diffD = trailingSuspendDays.diff().fillna(0)
        empty = pd.DataFrame(0, index = isSuspend.index, columns = isSuspend.columns)
        empty[diffD < -longSuspendDays] = 1
        arr = empty.values
        x , y= np.where(arr == 1)
        for i in range(1, backDays):
            arr[(x + i, y)] = 1
        
        arr = pd.DataFrame(arr)
        arr.index = isSuspend.index
        arr.columns = isSuspend.columns
        return arr

    
    @staticmethod
    def getCodeSep(df):
        kc_lst = []
        cy_lst = []
        other = []
        for i in df.columns:
            code, market = i.split('.')
            if market == 'SH' and int(code[:3]) >= 688:
                kc_lst.append(i)
            
            elif market == 'SZ' and int(code[:3]) >= 300 and int(code[:3]) < 500:
                cy_lst.append(i)
            else:
                other.append(i)
            
        return kc_lst, cy_lst, other
     


    @staticmethod
    def available(isSuspend, isST, isNew, isRecover):
        
        isSuspend = isSuspend.fillna(1).astype(bool)
        isST = isST.fillna(1).astype(bool)
        isNew = isNew.fillna(1).astype(bool)
        isRecover = isRecover.fillna(1).astype(bool)
        
        return ~ (isSuspend|isST|isNew|isRecover)
    
    @staticmethod
    def getAvailableStockUniverse(
                                  dataloader,
                                  newDays = (252, 252, 63), 
                                  suspendDays = 63, 
                                  recoverDays = 5):
        isST = dataloader.getST().fillna(1).astype(bool)
        ListedDays = dataloader.getListedDays()
        kcCode, cyCode, otherCode = StockUniverse.getCodeSep(ListedDays)
        kcDays, cyDays, otherDays = newDays
        isNew = pd.concat([ListedDays.fillna(-1)[kcCode] < kcDays, 
                           ListedDays.fillna(-1)[cyCode] < cyDays,
                           ListedDays.fillna(-1)[otherCode] < otherDays], axis = 1).sort_index(axis = 1)
        isSuspend = dataloader.getSuspend()
        isRecover = StockUniverse.getLongSuspendTF(isSuspend, suspendDays, recoverDays)
        universe = StockUniverse.available(isSuspend, isST, isNew, isRecover)
        for i in universe.columns:
            if i.split('.')[1] == 'BJ':
                universe.loc[:, i] = False
        return universe
    
    
    @staticmethod 
    def getAvailableStockUniverseWithoutSPZDT(dataloader, 
                                            **kargs):
        
        spzt = dataloader.getSPZT().fillna(True)
        spdt = dataloader.getSPDT().fillna(True)
        zdt = spzt | spdt
        universe = StockUniverse.getAvailableStockUniverse(
            dataloader, **kargs)
        return universe & (~zdt)
        


class SolveBarraToolBox(BarraToolBox):
    
    
    def __init__(self, dataloader, barraloader, useRiskFactor, availableUniverse,
                 njobs = 16, method = 'solveF', start = '2015-01-01', 
                 alpha_factors = [], alpha_names = []):
        
        self.dataloader = dataloader
        self.barraloader = barraloader
        self.useRiskFactor = useRiskFactor
        self.njobs = njobs
        self.method = method
        self.start = start
        self.alpha_factors = alpha_factors
        self.alpha_names = alpha_names
        self.availableUniverse = availableUniverse
        if len(alpha_factors) > 0 and len(self.alpha_names) == 0:
            self.alpha_names =  [f'alpha{i}' for i in range(len(alpha_factors))]
        
        
    
    def cleanPrep(self):
        del self.dic
    
    def cleanWeight(self):
        del self.weight_dict
    
    def cleanFactorReturn(self):
        del self.factor_return
    
    def close(self):
        self.cleanPrep()
        self.cleanWeight()
        self.cleanFactorReturn()
        
    
    
    def getPreparedDict(self):
        market_size = self.dataloader.getCirculateMarketValue()
        next_period_stockret = self.dataloader.getRealizedNextPeriodReturn()
        industry = self.dataloader.getIndustry()
        
        if len(self.alpha_factors)==0:    
            res = utils._align(self.availableUniverse, market_size, next_period_stockret, industry, *[self.barraloader.getFinalFactor(i) for i in self.useRiskFactor])
        else:
            res = utils._align(self.availableUniverse, market_size, next_period_stockret, industry, *[self.barraloader.getFinalFactor(i) for i in self.useRiskFactor], *self.alpha_factors)
        res = [i[self.start:] for i in res]
        self.availableUniverse = res[0]
        size = res[1]
        stockRet = res[2]
        industry = res[3]
        factors  = res[4:]
    
    
        self.dic = {}
        
        
        for i in tqdm(range(len(stockRet)), desc = '准备数据'):
            r = stockRet.iloc[i,:]
            ii = industry.iloc[i,:]
            s = size.iloc[i,:]
            f = [ff.iloc[i,:] for ff in factors]
            a = self.availableUniverse.iloc[i,:]
            df = pd.concat([r, ii] + f, axis = 1).dropna()
            if len(df) == 0:
                continue
            
            constrained = a[a == 1].index
            df = df.loc[set(constrained).intersection(set(df.index)), :]
            y = df.iloc[:,0]
            ind = pd.get_dummies(df.iloc[:,1]).sort_index(axis = 1)
            fac = df.iloc[:,2:]
            fac.columns = self.useRiskFactor + self.alpha_names
            x = pd.concat([ind, fac], axis = 1)
            
            weight = s.loc[x.index]
            
            temp = pd.concat([s, ii],axis = 1).loc[x.index]
            temp.columns= ['weight', 'ind']
            ind_size = temp.groupby('ind').mean().sort_index()['weight']
    
    
            self.dic[stockRet.index[i]] = {
                    'factors': x.sort_index(),
                    'return' : y.sort_index(),
                    'cap_weight' : weight,
                    'ind_weight': ind_size.sort_index()
                    }
    #%%
    '''
    收益率
    '''
    @staticmethod
    def getLinearConstrainMatrix(factor, ind_size):

        ind_num = len(ind_size)
        ind_size = np.array(ind_size)
        num_risk_factors = factor.shape[-1] - len(ind_size) - 1
        reduced_weights = -ind_size[:-1]/ind_size[-1]
        upperleft = np.row_stack((np.eye(ind_num), np.append([0], reduced_weights)))
        res = np.zeros(shape = (ind_num + num_risk_factors + 1, ind_num + num_risk_factors))
        res[: (ind_num + 1), :ind_num] = upperleft
        res[(ind_num + 1):, ind_num:] = np.eye(num_risk_factors)
    

        return res
    
    
    @staticmethod
    def solveFactorReturn(dic):
        factor = sm.add_constant(dic['factors']).values
        ind_size = dic['ind_weight'].values
        weight = dic['cap_weight'].values
        weight = np.sqrt(weight)/np.sqrt(weight).sum()        
        
        C = SolveBarraToolBox.getLinearConstrainMatrix(factor, ind_size)
        b = factor
        weight = np.array(weight)
        W = np.diag(weight/weight.sum())
        
        factorWeight = C.dot( np.linalg.inv(C.T.dot(b.T).dot(W).dot(b).dot(C)) ).dot(C.T).dot(b.T).dot(W)
        res = pd.DataFrame(factorWeight)
        res.index = sm.add_constant(dic['factors']).columns
        res.columns =  dic['factors'].index

        return res.T            
        
    
    @staticmethod
    def solvePureFactorReturn(dic):
        factor = dic['factors'].values
        ind_size = dic['ind_weight'].values
        weight = dic['cap_weight'].values
        weight = np.sqrt(weight)/np.sqrt(weight).sum()
        #    weight = weight/weight.sum()
        
        industry_num = len(ind_size)
        style_factor_matrix = factor[:, industry_num :]
#        industry_matrix = factor[:, : industry_num]
        available_num_stocks = factor.shape[0] 
        lst = []
        for i in range(style_factor_matrix.shape[1]):        
            w = cp.Variable(available_num_stocks)            
            resid_style_factor = np.delete(style_factor_matrix, i, axis = 1)
            self_style_factor = style_factor_matrix[:, i]
            
        #        constraints = [    
        ##                           w.T @ self_style_factor >= 0.9,
        ##                           w.T @ self_style_factor <= 1,
        ##                           w.T @ resid_style_factor <= 0.1,
        ##                           w.T @ resid_style_factor >= -0.1,
        #                           (w.T - weight.T)@resid_style_factor == 0,
        #                           w >= 0,
        #                           w <= 0.005,
        ##                           w.T @industry_matrix <= (ind_size/ind_size.sum()) + 0.05,
        ##                           w.T @industry_matrix >= np.clip((ind_size/ind_size.sum()) - 0.05, 0, np.inf),
        #                           cp.sum(w) == 1
        #                           ]
        
            step = 0.0001
            success = False
            for j in range(20):
                constraints = [    
                                   (w.T - weight.T)@resid_style_factor == 0,
                                   w >= 0,
                                   w <= 0.003 + step * j,
                                   cp.sum(w) == 1
                                   ]
                obj = cp.Maximize(w.T @ self_style_factor)
                prob = cp.Problem(obj, constraints)
                
                try:
                    prob.solve(solver = 'ECOS', verbose = False)
                    if w.value is None:
                        continue
                    append_val = w.value
                    success = True
                    break
                except:
                    continue
        
                        
            if not success:
                append_val = np.ones(available_num_stocks) * np.nan
            if w.value is None:
                append_val = np.ones(available_num_stocks) * np.nan
            
            lst.append(append_val)
        
        
        res = pd.DataFrame(lst)
        res.index =  dic['factors'].columns[industry_num:]
        res.columns =  dic['factors'].index
        res = res.round(6)
        res[res <0] = 0
        return res.T    
    
    

    
    
    def getWeight(self):
    
        self.weight_dic = {}
        if self.method == 'solveF':
            func = SolveBarraToolBox.solveFactorReturn
        elif self.method == 'solvePureF':
            func = SolveBarraToolBox.solvePureFactorReturn
        
        if not hasattr(self, 'dic'):
            self.getPreparedDict()
        
        job = self.njobs if self.method =='solvePureF' else 1
        
        if job == 1:
            temp = []
            for t in tqdm(self.dic.keys(), 
                          desc = '计算因子权重', 
                          total = len(list(self.dic.keys())) ):
                temp.append(func(self.dic[t]) )            
                
            
        else:
        
            with utils.tqdm_joblib(tqdm(desc = '计算因子权重', 
                                        total = len(list(self.dic.keys())))):    
            
                temp = joblib.Parallel(job)(joblib.delayed(func)(self.dic[t]) 
                                            for t in self.dic.keys())
            
        for i, t in enumerate (self.dic.keys() ):
            self.weight_dic[t] = temp[i]
        
        
        return self.weight_dic
    
    def getFactorRet(self):
    
        ret = []
        if not hasattr(self, 'weight_dic'):
            self.getWeight()
            
        for i in tqdm(self.weight_dic.keys(),
                      desc = '计算因子收益率',
                      total = len(list( self.weight_dic.keys()) )) :
            weight = self.weight_dic[i]
            data = self.dic[i]
            if self.method == 'solveF':
                ret.append(pd.Series(weight.values.T.dot(data['return']).reshape(-1),
                                     index = weight.columns, name = i))
            elif self.method == 'solvePureF':
                index_weights = (np.sqrt(data['cap_weight'])/np.sqrt(data['cap_weight']).sum()).values            
                exposure = (weight.values.T - index_weights).dot(data['factors'].iloc[:, len(data['ind_weight']): ])
                excessed_exposure = np.diag(1/np.diagonal(exposure))
                res = excessed_exposure.dot((weight.values.T - index_weights).dot(data['return']))
                ret.append(pd.Series( np.hstack(  (  np.array([0] *  (len(data['ind_weight']) + 1) )   ,res.reshape(-1))  )   ))
        
        ret = pd.concat(ret, axis = 1)
        
        
        self.factor_return = ret.T
        return ret.T

    def getLinearReturnPrediction(self, rollingDays = 120):
        if not hasattr(self, 'factor_return'):
            self.getFactorRet()
        estimated_factor_return = self.factor_return.shift(1).rolling(window = rollingDays).mean().dropna()
        lst = []
        for time in estimated_factor_return.index:
            X = self.dic[time]['factors']
            if 'const' not in X.columns:
                X = sm.add_constant(X)
            return_1d_forecast = X.dot(estimated_factor_return.loc[time])
            lst.append(return_1d_forecast)
        
        forecast_return = pd.concat(lst, axis = 1).T
        forecast_return.index = estimated_factor_return.index
        return forecast_return
    
    
#%%    
    '''
    协方差矩阵预测
    '''
    def getCovDict(self, cov_estimate_window = 252, cov_half_life = 90,
                         vol_estimate_window = 252, vol_half_life = 90,
                         neweywest_vol_lag = 2, neweywest_cov_lag = 2):
        if not hasattr(self, 'factor_return'):
            self.getFactorRet()
        weight = utils.getExponentialWeight(window = cov_estimate_window, half_life = cov_half_life)
        self.cov_dict = utils.fastRollingCov(self.factor_return, cov_estimate_window, weight)
        if neweywest_cov_lag == 'auto':
            neweywest_cov_lag = round(4 * (cov_estimate_window/ 100) ** (2/9))
        
        auto_cov_dict_lst = [utils.fastRollingAutoCov(self.factor_return, cov_estimate_window, weight, i) for i in range(1, 1 + neweywest_cov_lag)]
        self.neweywest_cov_dict = SolveBarraToolBox.neweyWestAdjustment(self.cov_dict, auto_cov_dict_lst)
        
        var_weight = utils.getExponentialWeight(window = vol_estimate_window, half_life = vol_half_life)
        varEstimateDF = utils.fastRollingVar(self.factor_return, vol_estimate_window, var_weight).dropna()
        
        if neweywest_vol_lag == 'auto':
            neweywest_vol_lag = round(4 * (vol_estimate_window/ 100) ** (2/9))        
        autovarEstimate_lst = [utils.fastRollingAutoVar(self.factor_return ,vol_estimate_window, var_weight, i).dropna() for i in range(1, 1 + neweywest_vol_lag)]
        self.neweywest_var_df = SolveBarraToolBox.neweyWestAdjustmentSeries(varEstimateDF, autovarEstimate_lst)
        self.volAdjustOnCov()
    



        
    
    
    
    
    def volAdjustOnCov(self):
        self.vol_adjusted_neweywest_cov_dict = {}
        for time in self.neweywest_cov_dict.keys():
            covmat = self.neweywest_cov_dict[time].values
            var = self.neweywest_var_df.loc[time].values
            
            orig = np.diag(1/np.sqrt(np.diag(covmat)))
            new = np.diag(np.sqrt(var))
            
            origcormat = orig.dot(covmat).dot(orig)
            newcovmat = new.dot(origcormat).dot(new)
            self.vol_adjusted_neweywest_cov_dict[time] = pd.DataFrame(newcovmat, index = self.neweywest_cov_dict[time].index, columns = self.neweywest_cov_dict[time].columns)
    
    
    def eigenAdjustment(self, cov_dict, empirical_factor = 1.2,
                        sample_len = 10000, sample_size = 252, 
                        seed = 42):
        exec_num = len(list(cov_dict.keys()))
        seed_lst = np.array(list(range(exec_num))) + seed
        with utils.tqdm_joblib(tqdm(desc = '特征根调整', total = exec_num ) ) :    
            temp = joblib.Parallel(self.njobs)(joblib.delayed(SolveBarraToolBox.eigenAdjustmentforDF)(
                    cov_dict[t], 
                    empirical_factor = empirical_factor,
                    sample_len = sample_len, sample_size = sample_size, seed = s) 
                                for t, s in zip(cov_dict.keys(),seed_lst) 
                                )
        self.eigen_adjusted_vol_dict = dict(zip(cov_dict.keys(), temp ))
    
    
    
    
    
    
    
    
    @staticmethod
    def neweyWestAdjustment(S0_dict, Sn_dict_lst):
        neweywestAdjust_dict = {}
        for time in S0_dict.keys():
            S0 = S0_dict[time]
            S_lst = [i[time] + i[time].T for i in Sn_dict_lst]
            q = len(S_lst)
            weight = [1 - i/(1 + q) for i in range(1, q + 1)]    
            for mat, w in zip(S_lst, weight):
                S0 = S0 + w * mat    
            
            neweywestAdjust_dict[time] = S0 
        return neweywestAdjust_dict                
                
    @staticmethod
    def neweyWestAdjustmentSeries(S0, Sn_lst):
        q = len(Sn_lst)
        weight = [1 - i/(1 + q) for i in range(1, q + 1)]    
        for num, w in enumerate(weight):
            S0 = S0 + (Sn_lst[num] + Sn_lst[num])*w
        
        return S0          
        
    @staticmethod
    def eigenAdjustmentforDF(fcovdf,
                        empirical_factor = 1.2,
                        sample_len = 10000, sample_size = 252, seed = 42):
        fcov = fcovdf.values
        factor_size = fcov.shape[1]
        w, v = np.linalg.eig(fcov)
        np.random.seed(seed)
        noise = np.random.normal(size = [factor_size, sample_size, sample_len])
        scaled_noise = noise * np.sqrt(w).reshape(-1, 1, 1)
        
#        weight = utils.getExponentialWeight(252, 90)
#        lag = 2
#        bias_v = np.sqrt(np.mean(utils.fastEigenSampling2(v, scaled_noise, fcov, weight, lag), axis = 0))
        bias_v = np.sqrt(np.mean(utils.fastEigenSampling(v, scaled_noise, fcov), axis = 0))
        bias_v = empirical_factor*(bias_v - 1) + 1
        Dnew = np.diag(bias_v ** 2) * np.diag(w)
        adjusted_fcov = v.dot(Dnew).dot(v.T)
        return pd.DataFrame(adjusted_fcov, index = fcovdf.index, columns = fcovdf.columns)
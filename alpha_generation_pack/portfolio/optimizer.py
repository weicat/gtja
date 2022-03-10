import pandas as pd 
import cvxpy as cp
import numpy as np 
import datetime
import zipfile
import os

class Optimizer(object):
    pass

    @staticmethod
    def manageIOStockList(lastWeight_, riskFactor, returnForecast, 
                          thisIndexWeight, thisSuspend):
        #把指数权重, 上一期权重，预测都放到一起
        #放到一起是因为不想考虑全市场，变量能少一点是一点
        lastWeight = lastWeight_.copy()
        lastWeight = lastWeight[lastWeight!=0]
        indexComponents = thisIndexWeight[thisIndexWeight !=0 ]
        allConcat = pd.concat([lastWeight, returnForecast, indexComponents], axis = 1).sort_index()
        modifiedLastWeight, modifiedReturnForecast, modifiedIndexComponents = \
                    allConcat.iloc[:, 0], allConcat.iloc[:, 1], allConcat.iloc[:, 2]
        
        
        #得到相应的风险因子，风险因子因为是对全市场的，
        #所以没放在一起concat
        modifiedFactor = riskFactor.copy()
        modifiedFactor.loc[set(allConcat.index) - set(modifiedFactor.index),:] = np.nan
        modifiedFactor = modifiedFactor.loc[allConcat.index]
    
        #上一期没有的权重，这一期有了，无论是因为新加了forecast还是指数的index
        #都应该填成0
        modifiedLastWeight = modifiedLastWeight.fillna(0)
        #和上面同理
        modifiedIndexComponents = modifiedIndexComponents.fillna(0)
        
        #下面这两个可以讨论怎么改，暂定都是0
        #上一期有权重或者有指数权重,这一期没有forecast/risk
        #可能的原因大概是 如果forecast的是量价因子，那就是停牌/退市了，或者变成ST了，
        #那直接填0表明 中性立场
        modifiedReturnForecast = modifiedReturnForecast.fillna(0)
        modifiedFactor = modifiedFactor.fillna(0)
    
        #下面这段都是想找出股票是退市了还是没退
        #如果没退市, 就继续持有，如果退市了，就直接权重变成0，所以要分开处理
        #不想再引入一个新的变量，即所有的股票，如果不在停牌股中，我就不要了全把weight = 0，
        #包括的情况可能就是退市/变成ST，不停牌我就可以交易，退市的股票再dataloader的return中是会变成-1的
        thisSuspendDropped = thisSuspend.dropna().astype(bool)    
        suspendIndex = set(thisSuspendDropped[thisSuspendDropped == True].index)
        #returnforecast代表两个信息，一个是值的大小，一个是我今天选出来要挑的stockUniverse
        #这个其实就是上一期 + index 比 returnforecast多的股票，警惕一下这些停牌股不能动
        leaveOutIndex = set(allConcat.index) - set(returnForecast.index)
        suspendIndexPos = []
        otherPos = []
        for i in leaveOutIndex:
            pos = list(allConcat.index).index(i)
            if i in suspendIndex:
                suspendIndexPos.append(pos)
            else:
                otherPos.append(pos)
                
        return (modifiedLastWeight, modifiedReturnForecast, modifiedIndexComponents, modifiedFactor),\
                (suspendIndexPos, otherPos)



class BarraOptimizer(Optimizer):
    '''
    my default optimizer
    '''
    
    
    
    omit_stocks = {
        '600068.SH' : '2021-09-01',
        '600723.SH': '2021-09-15'

        }
    
    benchmark_index = '000905.SH'
    def __init__(self, dataloader, barraloader):
        self.dataloader = dataloader
        self.barraloader = barraloader
        self.index_weight = self.dataloader.getIndexWeight(self.benchmark_index)
        
        self.opt_params  = {
            
            'risk_passive_relative_rest' : 0.1,
            'ind_passive_relative_rest' : 0.1,
            'weight_passive' :0.02,
            'round_turnover': 0.2
            
            
            }
        
        
    def setOptParams(self, dic):
        self.opt_params = dic
    
    def getRiskFactors(self, time):
        
        ind = pd.get_dummies(self.dataloader.getIndustry().loc[time])
        self.industry_name = ind.columns        
        barra = [self.barraloader.getFinalFactor(n).loc[time] for n in self.barraloader.style_factor_name]
        self.style_name = self.barraloader.style_factor_name
        barra = pd.concat(barra, axis = 1)
        barra.columns = self.style_name
        df = pd.concat([ind, barra], axis = 1)
        
        df = df.fillna(0)
        return df
    
    def getSuspend(self, time):
        return self.dataloader.getSuspend().loc[time]
    
    
    def getIndexWeight(self, time):
        t = self.index_weight.loc[time]
        t = t.dropna()
        t = t/t.sum()
        return t
        
    def checkOmit(self, modifiedFactor, time):
        lst = []
        for stock in self.omit_stocks:
            omit_time = pd.to_datetime(self.omit_stocks[stock])
            if pd.to_datetime(time) >= pd.to_datetime(omit_time):
                if stock in modifiedFactor.index:
                    lst.append(list(modifiedFactor.index).index(stock))
        
        return lst
    
    def kickAndRecordOmitArr(self):
        lst = self.omit_arr
        self.zt_arr = list(set(self.zt_arr) - set(lst) )
        self.dt_arr = list(set(self.dt_arr) - set(lst) )
        self.suspended_index = list(set(self.suspended_index) - set(lst))
        self.controlled_unavailable_index = list(
            set(self.controlled_unavailable_index).union(set(lst)))
    
    def setZDTArr(self, time, chosen_stocks):
        zt = self.dataloader.getSPZT().loc[time].fillna(False)
        dt = self.dataloader.getSPDT().loc[time].fillna(False)
        zt.loc[set(chosen_stocks) - set(zt.index)] = False
        dt.loc[set(chosen_stocks) - set(dt.index)] = False
        self.thiszt = zt.loc[chosen_stocks]
        self.thisdt = dt.loc[chosen_stocks]

        self.zt_arr = np.where(self.thiszt == True)[0]
        self.dt_arr = np.where(self.thisdt == True)[0]
        
    def setPrep(self, iterate, last_weight, forecast, time):
        self.iterate = iterate
        risk_factors = self.getRiskFactors(time)
        this_index_weight = self.getIndexWeight(time)
        this_isSuspend = self.getSuspend(time)
     
        #进优化器前先把所有的可能有停牌/退市的地方都写好    
        (self.weight_before, self.this_forecast_return, 
         self.this_index_weight, modifiedFactor),\
         (self.suspended_index, self.elseunavailable_index) = \
         Optimizer.manageIOStockList(last_weight, risk_factors, forecast, this_index_weight, this_isSuspend)
         
        #设置股票数量、涨跌停
        self.style_factor_matrix = modifiedFactor[self.style_name]
        self.industry_matrix = modifiedFactor[self.industry_name]
        self.available_num_stocks = modifiedFactor.shape[0]        
        self.setZDTArr(time, modifiedFactor.index)
    
        #会和买卖跌停对撞
        self.controlled_unavailable_index = []
        for elseindex in self.elseunavailable_index:
            if self.thiszt[elseindex] == False and self.thisdt[elseindex] == False:
                self.controlled_unavailable_index.append(elseindex)
        
        #一定要今天走的票，这个是手动设置的，一般是股票要退市转成另一个股了
        self.omit_arr = self.checkOmit(modifiedFactor, time)
        self.kickAndRecordOmitArr()    
        
        self.w = cp.Variable(self.available_num_stocks)
    
    def setObjecitve(self):
        '''
        w, myforecast
        '''
        self.obj = cp.Maximize((self.w.T) @ self.this_forecast_return )
    
    def createBaseConstraint(self):
        
        self.constraints = [self.w >= 0,
        cp.sum(self.w) == 1    
        ]
        if len(self.zt_arr) > 0:
            self.constraints.append(self.w[self.zt_arr] <= self.weight_before[self.zt_arr])
        if len(self.dt_arr) > 0:
            self.constraints.append(self.w[self.dt_arr] >= self.weight_before[self.dt_arr])
        if len(self.suspended_index) > 0:
            self.constraints.append(self.w[self.suspended_index] == self.weight_before[self.suspended_index])
        if len(self.controlled_unavailable_index) > 0:
            self.constraints.append(self.w[self.controlled_unavailable_index] == 0)
        
    
    def addRiskConstraints(self):
        self.addMatrixProductConstraint(self.style_factor_matrix, 'risk')
    def addIndustryConstraints(self):
        self.addMatrixProductConstraint(self.industry_matrix, 'ind')
    
    
    

    
    def addMatrixProductConstraint(self, mat, spec_name):
        t = list(self.opt_params.keys())
        passive_relative_rest_flag = False
        passive_gross_rest_flag = False
        active_gross_rest_flag = False
 
        passive_relative_lst = []
        passive_gross_lst = []
        active_gross_lst = []      
        
        for i in t:
            val = self.opt_params[i]

            if spec_name in i :
                _, style, method, factor_name = i.split('_')
                
                if style == 'passive':
                    if method == 'relative':
                        if factor_name != 'rest':
                            
                            if not val is None: 
                                self.constraints.append(
                                cp.abs((self.w.T - self.this_index_weight.T) @ \
                                       mat[factor_name]) <= \
                                       val* cp.abs(self.this_index_weight.T @ mat[factor_name]))
                            
                            passive_relative_lst.append(factor_name)
                        else:
                            passive_relative_rest_flag = True
                            passive_relative_rest_val = val
                    elif method == 'gross':
                        if factor_name != 'rest':
                            
                            if not val is None: 
                                self.constraints.append(
                                cp.abs((self.w.T - self.this_index_weight.T) @ \
                                       self.mat[factor_name]) <= \
                                       val)
                            passive_gross_lst.append(factor_name)
                            
                        else:
                            passive_gross_rest_flag = True
                            passive_gross_rest_val = val
                        
                    
                elif style == 'active':
                    if method == 'relative':
                        raise ValueError('active portfolio style do not have a relative method')
                    elif method == 'gross':
                        if factor_name != 'rest':
                            
                            if not val is None: 
                                self.constraints.append(
                                cp.abs(self.w.T @ \
                                       mat[factor_name]) <= \
                                       val)
                            active_gross_lst.append(factor_name)
                            
                        else:
                            active_gross_rest_flag = True
                            active_gross_rest_val = val        
        
        if passive_relative_rest_flag:
            temp_factors = set(mat.columns) - set(passive_relative_lst)
            self.constraints.append(
            cp.abs((self.w.T - self.this_index_weight.T) @ \
                   mat[temp_factors]) <= \
                   passive_relative_rest_val* cp.abs(self.this_index_weight.T @ mat[temp_factors]))
            
        if passive_gross_rest_flag:
            temp_factors = set(mat.columns) - set(passive_gross_lst)
            self.constraints.append(
            cp.abs((self.w.T - self.this_index_weight.T) @ \
                   mat[temp_factors]) <= \
                   passive_gross_rest_val)
        
        if active_gross_rest_flag:
            temp_factors = set(mat.columns) - set(active_gross_lst)
            self.constraints.append(
            cp.abs(self.w.T @ \
                   mat[temp_factors]) <= \
                   active_gross_rest_val)
            
            
    def addTurnOverConstraints(self):
        t = list(self.opt_params.keys())
        for i in t:
            if 'round_turnover' in i:
                val = self.opt_params[i]
                if self.iterate != 0:
                    self.constraints.append(cp.sum(cp.abs(self.w - self.weight_before)) <= val)        
            
    
    def addWeightConstraints(self):
        t = list(self.opt_params.keys())
        for i in t:
            if 'weight' in i:
                val = self.opt_params[i]
                _, method = i.split('_')
                if method == 'active':
                    self.constraints.append(
                        cp.abs(self.w.T) <= val,
                        )
                elif method == 'passive':
                    self.constraints.append(
                        cp.abs(self.w.T - self.this_index_weight.T) <= val
                        )
        
        
        
        
    
    def setConstraint(self):
        
        
        self.createBaseConstraint()
        self.addRiskConstraints()
        self.addIndustryConstraints()
        self.addTurnOverConstraints()
        self.addWeightConstraints()
        
        # elif self.mode ==2:
        # self.constraints = [
        #     self.w >= 0,
        #     cp.sum(self.w) == 1  
        #     ]
        
        # if len(self.zt_arr) > 0:
        #     self.constraints.append(self.w[self.zt_arr] <= self.weight_before[self.zt_arr])
        # if len(self.dt_arr) > 0:
        #     self.constraints.append(self.w[self.dt_arr] >= self.weight_before[self.dt_arr])
        # if len(self.suspended_index) > 0:
        #     self.constraints.append(self.w[self.suspended_index] == self.weight_before[self.suspended_index])
        # if len(self.controlled_unavailable_index) > 0:
        #     self.constraints.append(self.w[self.controlled_unavailable_index] == 0)
        
        # self.constraints.append(
        #         cp.abs((self.w.T - self.this_index_weight.T) @ self.style_factor_matrix) <= \
        #                                             0.1 * cp.abs(self.this_index_weight.T @ self.style_factor_matrix))
                
        # self.constraints.append(    
        #     cp.abs((self.w.T - self.this_index_weight.T) @ self.industry_matrix) <= \
        #                                             0.1 * cp.abs(self.this_index_weight.T @ self.industry_matrix))
            
        # if self.iterate != 0:
        #     self.constraints.append(cp.sum(cp.abs(self.w - self.weight_before)) <= 0.2)
            
        # self.constraints.append(cp.abs(self.w.T - self.this_index_weight.T) <= 0.02)
            

        
            
    def cast(self, wnew):
        if len(self.zt_arr) > 0:
            wnew[self.zt_arr] = wnew[self.zt_arr].clip(max = self.weight_before[self.zt_arr])
        if len(self.dt_arr) > 0:
            wnew[self.dt_arr] = wnew[self.dt_arr].clip(min = self.weight_before[self.dt_arr])
        if len(self.suspended_index) > 0:
            wnew[self.suspended_index] = self.weight_before[self.suspended_index]
        if len(self.controlled_unavailable_index) > 0:
            wnew[self.controlled_unavailable_index] = 0
        
        lst = []
        else_lst = []
        for i in range(len(wnew)):
            if i in self.zt_arr or i in self.dt_arr or i in self.suspended_index or\
                i in self.controlled_unavailable_index:
                    else_lst.append(i)
            else:
                lst.append(i)
        if len(else_lst) !=0:
            welsesum = np.sum(wnew[else_lst])
        else:
            welsesum = 0
        if len(lst) > 0:
            wnew[lst] = wnew[lst]/wnew[lst].sum() * (1 - welsesum)
        
        wnew = pd.Series(wnew, index = self.weight_before.index).round(6)
        wnew = wnew.clip(lower = 0)
        
            
        return wnew
    def optimize(self, iterate, last_weight, forecast, time):
        
        self.setPrep(iterate, last_weight, forecast, time)
        self.setObjecitve()
        self.setConstraint()
        
        
        prob = cp.Problem(self.obj, self.constraints)
        
        try:
            prob.solve(solver = 'ECOS', verbose = False)
            if self.w.value is None:
                raise ValueError('invalid_weights')
        except:
            try:
                prob.solve(solver = 'SCS', verbose = False)
                if self.w.value is None:
                    raise ValueError('invalid_weights')
            except:
                try:
                    prob.solve(solver = 'OSQP', verbose = False)
                    if self.w.value is None:
                        print('Cannot Solve Weight@' +  str(time))
                    self.w.value = self.weight_before.values
                except:
                    print('Cannot Solve Weight@' +  str(time))
                    self.w.value = self.weight_before.values
        

        wnew = self.cast(self.w.value) 
        # wnew = self.w.value

        wnew.name = time
        
        return wnew
        


class OnlineBarraOptimizer(BarraOptimizer):
    '''
    指数成分股换了要重新算
    现在的逻辑是用天软的昨天的非零权重股作为今天的成分股
    '''
    csi_official_path = 'Z:\LuTianhao\中证公司指数清单'
    benchmark_index = '000905.SH'
    iterate = 1
    def __init__(self, 
                 dataloader, 
                 barraloader, 
                 time):
        self.dataloader = dataloader
        self.barraloader = barraloader
        self.time = time
    
        self.setRiskFactors()
        self.setIndexWeight()
   
    
   
    def cast(self, wnew):
        
        if len(self.suspended_index) > 0:
            wnew[self.suspended_index] = self.weight_before[self.suspended_index]
        if len(self.controlled_unavailable_index) > 0:
            wnew[self.controlled_unavailable_index] = 0
        
        return wnew 
   
    def optimize(self, last_weight, forecast, this_isSuspend):
        
        self.setPrep(last_weight, forecast, this_isSuspend)
        self.setObjecitve()
        self.setConstraint()
        
        
        prob = cp.Problem(self.obj, self.constraints)
        
        try:
            prob.solve(solver = 'ECOS', verbose = False)
            if self.w.value is None:
                raise ValueError('invalid_weights')
        except:
            try:
                prob.solve(solver = 'SCS', verbose = False)
                if self.w.value is None:
                    raise ValueError('invalid_weights')
            except:
                prob.solve(solver = 'OSQP', verbose = False)
                if self.w.value is None:
                    print('Cannot Solve Weight@')
                self.w.value = self.weight_before.values       
        

        wnew = self.cast(self.w.value) 
        # wnew = self.w.value
        wnew = pd.Series(wnew, index = self.weight_before.index).round(6)
        wnew = wnew.clip(lower = 0)
        wnew = wnew/wnew.sum()
        wnew.name = 'Dollar Weight'
        
        return wnew
   
    
    
    
    def setRiskFactors(self):
        self.raw_risk_factors = self.getRiskFactors(self.time)        
    
    
    
    

    def setPrep(self, last_weight, forecast, this_isSuspend): 
        #进优化器前先把所有的可能有停牌/退市的地方都写好    
        (self.weight_before, self.this_forecast_return, 
         self.this_index_weight, modifiedFactor),\
         (self.suspended_index, self.elseunavailable_index) = \
         Optimizer.manageIOStockList(last_weight, self.raw_risk_factors, forecast, 
                                     self.raw_index_weight, this_isSuspend)
         
        #设置股票数量、涨跌停
        self.style_factor_matrix = modifiedFactor[self.style_name].sort_index() 
        self.industry_matrix = modifiedFactor[self.industry_name].sort_index()
        self.available_num_stocks = modifiedFactor.shape[0]        
    
        #会和买卖跌停对撞
        self.controlled_unavailable_index = []
        for elseindex in self.elseunavailable_index:
            self.controlled_unavailable_index.append(elseindex)
         
        self.w = cp.Variable(self.available_num_stocks)
    
    
    
    

    def setIndexWeight(self):
        df = self.getCSIWeight()
        bench = self.dataloader.getIndexWeight(self.benchmark_index).loc[self.time].dropna()
        df = df.set_index('成分券代码\nConstituent Code')['权重(%)\nWeight(%)']    
        new_arranged_index = []
        for i in df.index:
            t = str(i)
            t = '0' * (6 - len(t))  + t
            if int(i) < 600000:
                t += '.SZ'
            else:
                t += '.SH'
            new_arranged_index.append(t)
        
        df.index = new_arranged_index
        df = df.loc[bench[bench!=0].index]
        self.raw_index_weight = df/df.sum() 
            
        
        
    def getCSIWeight(self):
        time = self.time
        time = str(pd.to_datetime(time))[:10]
        time = ''.join(time.split('-'))
        pfx = '000906weightnextday'
        path_name = os.path.join(self.csi_official_path, pfx + time)
        if os.path.exists(path_name):
            df = pd.read_excel(os.path.join(path_name, 
                                            pfx + time +'.xls')
                               , index_col = 0)
            
            return df
        else:
            zippath = os.path.join(
                    self.csi_official_path, 
                    pfx + time + '.zip')
            if os.path.exists(zippath):
                zip_file = zipfile.ZipFile(zippath)
                for names in zip_file.namelist():
                    zip_file.extract(names, path_name)
                zip_file.close()
                self.getBenchMarkWeight(time)
            else:
                raise ValueError('先用ftp_download下一下')
    
    def setConstraint(self):
        '''
        w, this_index_weight
        style_factor_matrix,
        industry_matrix,
        zt_arr,
        dt_arr,
        suspend,
        
        '''
        self.constraints = [
                cp.abs((self.w.T - self.this_index_weight.T) @ self.style_factor_matrix) <= \
                                                    0.1 * cp.abs(self.this_index_weight.T @ self.style_factor_matrix),
                cp.abs((self.w.T - self.this_index_weight.T) @ self.industry_matrix) <= \
                                                    0.1 * cp.abs(self.this_index_weight.T @ self.industry_matrix),
                cp.abs(self.w.T - self.this_index_weight.T) <= 0.02,
                self.w >= 0,
                cp.sum(self.w) == 1    
                ]
            
        if self.iterate != 0:
            self.constraints.append(cp.sum(cp.abs(self.w - self.weight_before)) <= 0.2)
        
        if len(self.suspended_index) > 0:
            self.constraints.append(self.w[self.suspended_index] == self.weight_before[self.suspended_index])
        if len(self.controlled_unavailable_index) > 0:
            self.constraints.append(self.w[self.controlled_unavailable_index] == 0)
    
    
    
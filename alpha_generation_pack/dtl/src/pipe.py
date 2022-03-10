# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:07:50 2022

@author: Administrator
"""


import dtl.src.task as task
import numpy as np 
import copy
import datetime
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
import dtl.src.barra_factor as barra_factor
import os
import dtl.src.new_utils as utils
from tqdm import tqdm

class PipeLine(object):
    def __init__(self):
        pass



class BarraFactorComp(PipeLine):
    tasklist = [
        barra_factor.Size,
        barra_factor.Beta,
        barra_factor.RSTR,
        barra_factor.DaStd,
        barra_factor.CMRA,
        barra_factor.NonLinearSize,
        barra_factor.BooktoPrice,
        barra_factor.ETOP,
        barra_factor.STO,
        barra_factor.EGRF,
        barra_factor.GRO,
        barra_factor.LEV,
        barra_factor.DTOA
        ]
    
    compolist = [
        barra_factor.Size_Comp,
        barra_factor.Beta_Comp,
        barra_factor.Momentum_Comp,
        barra_factor.ResidualVolaility_Comp,
        barra_factor.NonLinearSize_Comp,
        barra_factor.BooktoPrice_Comp,
        barra_factor.Liquidity_Comp,
        barra_factor.EarningsYield_Comp,
        barra_factor.Growth_Comp,
        barra_factor.Leverage_Comp,
        
        ]
    
    skip_task = []
    def __init__(self, root_path, dataloader):
        self.root_path = root_path
        self.dataloader = dataloader
        self.initialize()
    
    
    
    def initialize(self):
        self.initialized_task_dict = {}
        for i in self.tasklist:
            self.initialized_task_dict[i.NAME] = i(os.path.join(self.root_path, 'descriptor'))
            
        self.initialized_comp_dict = {}
        for j in self.compolist:
            self.initialized_comp_dict[j.NAME] = j(os.path.join(self.root_path, 'factor'), 
                                                   *[self.initialized_task_dict[key]
                                                       for key in j.USED_FACTOR_DICT])

    def setCurrentTask(self, task_name, type_ = 'descriptor'):
        self.current_task_name = task_name
        if type_ == 'descriptor':
            self.current_task = self.initialized_task_dict[self.current_task_name]
        
        elif type_ == 'compofactor':
            self.current_task = self.initialized_comp_dict[self.current_task_name]

        
    def save(self):
        self.current_task.cal(self.dataloader)
        self.current_task.save()
    
    
    def run(self):
        time = datetime.datetime.now()
        print('Start Descriptor----------')
        self.runDescriptor()
        time_f = datetime.datetime.now()
        used_time = str(time_f.replace(microsecond = 0) - time.replace(microsecond = 0))
        print('Descriptor use time {t}'.format(t = used_time))
        
        print('Start CompoFactor---------')
        self.runCompoFactor()
        time_f = datetime.datetime.now()
        used_time = str(time_f.replace(microsecond = 0) - time.replace(microsecond = 0))
        print('CompoFactor use time {t}'.format(t = used_time))

        print('Start FinalFactor----------')
        self.runFinalFactor()
        time_f = datetime.datetime.now()
        used_time = str(time_f.replace(microsecond = 0) - time.replace(microsecond = 0))
        print('FinalFactor use time {t}'.format(t = used_time))
    
    
    def runDescriptor(self):
        for task_name in self.initialized_task_dict.keys():
            if task_name in self.skip_task:
                continue
            
            time = datetime.datetime.now()
            print('Start Task {t}'.format(t = task_name))
            self.setCurrentTask(task_name)
            self.save()
            time_f = datetime.datetime.now()
            used_time = str(time_f.replace(microsecond = 0) - time.replace(microsecond = 0))
            print('Task {t} finished. Used time {f}'.format(
                t = task_name, f = used_time))
    
    def runCompoFactor(self):
        
        for task_name in self.initialized_comp_dict.keys():
            if task_name in self.skip_task:
                continue
            
            time = datetime.datetime.now()
            print('Start Task {t}'.format(t = task_name))
            self.setCurrentTask(task_name, type_ = 'compofactor')
            self.save()
            time_f = datetime.datetime.now()
            used_time = str(time_f.replace(microsecond = 0) - time.replace(microsecond = 0))
            print('Task {t} finished. Used time {f}'.format(
                t = task_name, f = used_time))
        
    
    
    
    '''
    后面把这个弄进 factor里面
    '''
    def runFinalFactor(self):
        for i in self.initialized_comp_dict:
            if not hasattr(self.initialized_comp_dict[i], 'val'):
                self.initialized_comp_dict[i].load()
        
        
        time = datetime.datetime.now()
        print('Start Task Orthgonal')
        self.initialized_comp_dict['RESIDUALVOL_COMP'].val['ResidualVolatility'] = utils.orthogonal(
            self.initialized_comp_dict['RESIDUALVOL_COMP'].val['ResidualVolatility'], 
                    self.initialized_comp_dict['BETA_COMP'].val['Beta'], 
                    self.initialized_comp_dict['SIZE_COMP'].val['Size'])




        self.initialized_comp_dict['LIQUIDITY_COMP'].val['Liquidity'] = utils.orthogonal(
                self.initialized_comp_dict['LIQUIDITY_COMP'].val['Liquidity'],
                self.initialized_comp_dict['SIZE_COMP'].val['Size']
                )
        time_f = datetime.datetime.now()
        used_time = str(time_f.replace(microsecond = 0) - time.replace(microsecond = 0))
        print('Task Orthgonal finished. Used time {f}'.format(f = used_time))
        

        for i, v in tqdm(self.initialized_comp_dict.items(),
                         total = len(self.initialized_comp_dict.keys()),
                         desc = 'Calculating Final Factor'):
            if v.FUNDAMENTAL:
                v.fillIndustrial(self.dataloader)
            
            v.fillNA(self.dataloader)
            v.madWinsorize()
            v.weightedStandardize(self.dataloader)
            v.folder_path = os.path.join(self.root_path, 'Cal_factor')    
            v.save()
    

    
    
class CreateDataBase(PipeLine):


    tasklist  = [
        task.Financial,
        task.PV,
        task.IndexEOD,
        task.FreeFloat,
        task.IndexWeight,
        task.MarketCap,
        task.ST,
        task.SW1IND,
        task.SW1IND_NEW,
        task.STSheet,
        task.IndustryDictionary,
        task.Caps,
        task.SW1INDMERGED,
        task.PVProcess,
        task.IndexWeightProcess,
        task.IndexEODProcess,
        task.CDRMarketCap,
        task.CDRPV
        ]
    
    # tasklist = [
    #     task.CDRPV,
    #     task.PV,
    #     task.PVProcess,
    #     task.QUARTERBAR
        
    #     ]
    

    skip_task = []
    update_only_task = ['INDEXWEIGHT']
    def __init__(self, root_path, conn_dict):
        self.root_path = root_path
        self.conn = conn_dict
        self.initialize()
        
        
    @staticmethod
    def reverse(my_map):
        
        inv_map = {}
        for k, v in my_map.items():
            for j in v:
                inv_map[j] = inv_map.get(j, []) + [k]
                
        for i in my_map.keys():
            if i not in inv_map:
                inv_map[i] = []
                
        return inv_map
    
    def setDependencyDict(self):
        
        self.task_name = [j.NAME for j in self.tasklist]
        self.uninitialized_task_dict = dict(zip(self.task_name, self.tasklist))
        
        if len(np.unique(self.task_name)) != len(self.task_name):
               raise ValueError('Tasks have same names')
            
        
        
        self.forward_dict = {}
        for i in self.tasklist:
            temp = i.USED_TASK_LST
            if temp is None:
                self.forward_dict[i.NAME] = []
            else:
                self.forward_dict[i.NAME] = [j.NAME for j in temp]
                for t in self.forward_dict[i.NAME]: 
                    if t not in self.task_name:
                        raise ValueError(
                            '{t} is not in tasklist'.format(t = t))
        self.backward_dict = CreateDataBase.reverse(self.forward_dict)  
        
        if len(self.forward_dict.keys()) != len(self.backward_dict.keys()):
            raise ValueError('Check tasks')

        
        
    @staticmethod
    def topoSort(graph, names):
        
        
        in_degrees = dict((u,0) for u in names)   #初始化所有顶点入度为0
        vertex_num = len(in_degrees)
        for u in graph:
            for v in graph[u]:
                in_degrees[v] += 1       #计算每个顶点的入度
        Q = [u for u in in_degrees if in_degrees[u] == 0]   # 筛选入度为0的顶点
        Seq = []
        while Q:
            u = Q.pop()       #默认从最后一个删除
            Seq.append(u)
            for v in graph[u]:
                in_degrees[v] -= 1       #移除其所有指向
                if in_degrees[v] == 0:
                    Q.append(v)          #再次筛选入度为0的顶点
        if len(Seq) == vertex_num:       #如果循环结束后存在非0入度的顶点说明图中有环，不存在拓扑排序
            return Seq
        else:
            raise ValueError("there's a circle.")


    
    def setDependency(self):
        
        self.setDependencyDict()
        self.forward_dir = self.topoSort(self.forward_dict, self.task_name)
        self.backward_dir = self.topoSort(self.backward_dict, self.task_name) 
    
    def initialize(self):
        
        self.setDependency()
        self.check_close_dict = copy.deepcopy(self.backward_dict)
        self.initialized_task_dict = {}
        for task_name in self.backward_dir:
            i = self.uninitialized_task_dict[task_name]
    
            if i.USED_DATA_LST is None:
                conn = ''
            else:
                conn = self.conn[i.DB]
            
            if i.USED_TASK_LST is None:
                self.initialized_task_dict[i.NAME] = i(conn, 
                                                       root_path = self.root_path)
            else:
                args = [self.initialized_task_dict[j.NAME] for j in i.USED_TASK_LST]
                self.initialized_task_dict[i.NAME] = i(conn, *args,
                                                       root_path = self.root_path)
    

    def run(self):
        for task_name in self.backward_dir:
            if task_name in self.skip_task:
                continue
            
            time = datetime.datetime.now()
            print('Start Task {t}'.format(t = task_name))
            
            self.setCurrentTask(task_name)
            self.onLoopStart()
            self.onLoopEnd()
            time_f = datetime.datetime.now()
            used_time = str(time_f.replace(microsecond = 0) - time.replace(microsecond = 0))
            print('Task {t} finished. Used time {f}'.format(
                t = task_name, f = used_time))
    
    def setCurrentTask(self, task_name):
        self.current_task_name = task_name
        self.current_task = self.initialized_task_dict[self.current_task_name]
    
    def onLoopStart(self):
        try:
            if self.current_task_name in self.update_only_task:
                self.current_task.update()
            else:
                self.current_task.download()
        except:
            try:
                self.current_task.save()
            except:
                raise ValueError('Task {t} do not have a download method or save method '.format(
                    t = self.current_task_name))
        self.current_task.setDone()
        
    def onLoopEnd(self):
        self.close()
        
    def close(self):
        temp = copy.deepcopy(self.check_close_dict)
        for i in self.check_close_dict:
            if self.initialized_task_dict[i].getStatus():
                dependencies = self.check_close_dict[i]
                flag = True
                for dependency in dependencies:
                   if not self.initialized_task_dict[dependency].getStatus():
                       flag = False    
                       break
                if flag:
                    self.initialized_task_dict[i].close()
                    del temp[i]
        self.check_close_dict = temp
    
    
class UpdateDataBase(CreateDataBase):
     skip_task = ['INDUSTRY_DICTIONARY']
     def onLoopStart(self):
         try:
             self.current_task.update()
         except:
             try:
                 self.current_task.save()
             except:
                 raise ValueError('Task {t} do not have a download method or save method '.format(
                     t = self.current_task_name))
         self.current_task.setDone()
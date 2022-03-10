# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 10:09:43 2021

@author: Administrator
"""


import tensorflow as _tf
from keras.engine import compile_utils, data_adapter
from keras import backend
import numpy as np 
import datetime

class SimulatedAnnealingStrategy(_tf.keras.callbacks.Callback):
    
    '''
    这一版 eval loss 调用的是model 自己的evaluate，
    两个问题：
    1. 并不是fullbatch 的loss (fullbatch 试了，太慢)
    2. loss function 必须是train的loss 不能指定
    
    
    后续：
    一个快速的eval(直接用 val_loss)
    '''
    
    
    
    def __init__(self, 
                 init_temperature = 0.01, 
                 temperature_reduction_factor = 0.995,
                 step_lr_refresh = 4000, 
                 lr_reduction_factor = 0.8,
                 reduction_step = 1000,
                 min_delta = 0.0001,
                 min_lr = 0,
                 metric_name = 'loss',
                 eval_dataset = None,
                 save_best = True,
                 seed = 42):
    
        
        self.temperature = init_temperature
        self.temperature_reduction_factor = temperature_reduction_factor
        self.step_lr_refresh = step_lr_refresh
        self.lr_reduction_factor = lr_reduction_factor
        self.reduction_step = reduction_step
        self.min_delta = min_delta
        self.min_lr = min_lr
        


        self.save_best = save_best
        
    
        self.tot_iter = 0
        self.metric_name = metric_name
        
        
        # self.convert_eval_dataset(eval_dataset)
        self.random_state = np.random.RandomState(seed)
        self.x = eval_dataset       
        
        
    
    def check_refresh_learning_rate(self):
        if self.tot_iter % self.step_lr_refresh == 0:
            backend.set_value(self.model.optimizer.lr, self.restored_lr)

        
    def eval_loss(self):
        loss = self.model.evaluate(self.x,
                                   callbacks = [],
                                   verbose = False, 
                                   return_dict = True)[self.metric_name]
        return loss

    def savebest(self, now_loss):
        
        if self.save_best:
            if now_loss < self.loss_best:
                self.best_weight = self.model.get_weights()
            
            


    def check_simulate_anneal_process(self):
        if self.tot_iter % self.reduction_step == 0 :
            current_loss = self.eval_loss()
            self.temperature = self.temperature * self.temperature_reduction_factor
            if self.pre_loss - current_loss > self.min_delta:
                self.pre_weight = self.model.get_weights()
                self.savebest(current_loss)
            
            else:
                r = self.random_state.random()
                transit_prob = np.exp(-(current_loss - self.pre_loss)/self.temperature)
                if r > transit_prob:
                    self.model.set_weights(self.pre_weight)
                    backend.set_value(self.model.optimizer.lr, 
                                      backend.get_value(self.model.optimizer.lr) * \
                                          self.lr_reduction_factor
                                      )

                
                else:
                    self.pre_weight = self.model.get_weights()
                    
                    
    
    
    def on_train_begin(self, logs = None):
        

        self.restored_lr = backend.get_value(self.model.optimizer.lr)
        self.pre_loss = self.eval_loss()
        self.loss_best = self.pre_loss
        self.pre_weight = self.model.get_weights()
        self.best_weight = self.model.get_weights()


    def on_train_batch_begin(self, batch, logs=None):
        self.check_refresh_learning_rate()
        
        
        

    def on_train_batch_end(self, batch, logs=None):
        self.check_simulate_anneal_process()
        self.tot_iter += 1
        
        
        logs = logs or {}
        logs['Temperature'] = self.temperature
        logs['lr'] = backend.get_value(self.model.optimizer.lr)
    
    def on_epoch_end(self, epoch, logs=None):
        self.model.set_weights(self.best_weight)
        
    
    
class SimulatedAnnealingFast(_tf.keras.callbacks.Callback):
    
    '''
    这一版 eval loss直接调用的 val_loss, 算的更快
    '''
    
    
    
    def __init__(self, 
                 init_temperature = 0.01, 
                 temperature_reduction_factor = 0.995,
                 epoch_lr_refresh = 30, 
                 lr_reduction_factor = 0.8,
                 epoch_reduction = 2,
                 min_delta = 0.0001,
                 min_lr = 0,
                 metric_name = 'loss',
                 save_best = True,
                 seed = 42):
    
        
        self.temperature = init_temperature
        self.temperature_reduction_factor = temperature_reduction_factor
        self.epoch_lr_refresh = epoch_lr_refresh
        self.lr_reduction_factor = lr_reduction_factor
        self.epoch_reduction = epoch_reduction
        self.min_delta = min_delta
        self.min_lr = min_lr
        


        self.save_best = save_best       
        self.metric_name = metric_name
        
        
        # self.convert_eval_dataset(eval_dataset)
        self.random_state = np.random.RandomState(seed)
        
        
    
    def check_refresh_learning_rate(self, epoch):
        if epoch % self.epoch_lr_refresh == 0:
            backend.set_value(self.model.optimizer.lr, self.restored_lr)

        


    def savebest(self, now_loss):
        
        if self.save_best:
            if now_loss < self.loss_best:
                self.best_weight = self.model.get_weights()
            
            


    def check_simulate_anneal_process(self, epoch):
        if epoch % self.epoch_reduction == 0 :
            current_loss = self.current_loss
            self.temperature = self.temperature * self.temperature_reduction_factor
            if self.pre_loss - current_loss > self.min_delta:
                self.pre_weight = self.model.get_weights()
                self.savebest(current_loss)
            
            else:
                r = self.random_state.random()
                transit_prob = np.exp(-(current_loss - self.pre_loss)/self.temperature)
                if r > transit_prob:
                    self.model.set_weights(self.pre_weight)
                    backend.set_value(self.model.optimizer.lr, 
                                      backend.get_value(self.model.optimizer.lr) * \
                                          self.lr_reduction_factor
                                      )

                
                else:
                    self.pre_weight = self.model.get_weights()
                    
                    
    
    
    def on_train_begin(self, logs = None):
        

        self.restored_lr = backend.get_value(self.model.optimizer.lr)

    
    def on_epoch_end(self, epoch, logs=None):
        
        logs = logs or {}
        if epoch == 0:
            self.pre_loss = logs['val_loss']
            self.loss_best = logs['val_loss']   
            self.pre_weight = self.model.get_weights()
            self.best_weight = self.model.get_weights()
        
        else:
            self.current_loss = logs['val_loss']
            self.check_simulate_anneal_process(epoch)        
            self.check_refresh_learning_rate(epoch)
    
                
    
            logs['Temperature'] = self.temperature
            logs['lr'] = backend.get_value(self.model.optimizer.lr)
        
            self.model.set_weights(self.best_weight)
        
    


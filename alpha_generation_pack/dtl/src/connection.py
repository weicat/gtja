# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:22:29 2022

@author: Administrator
"""


import pymssql
import pyTSL



class Connection(object):
    pass


class WindDBData(Connection):
    
    def __init__(self):
    
        wind_db_address = '10.181.113.227'
        wind_password = 'z1tysp111@Gtja'
        wind_username = 'ztysp111'
        
        self.conn = pymssql.connect(wind_db_address,
                               wind_username,
                               wind_password,
                               charset = 'cp936')
        
        
        if self.conn is None:    
            raise ValueError('Cannot connect to WINDDB')
        else:
            print('------WINDDB Connected, Good to go------')
        
    
    def close(self):
        
        self.conn.close()
        print('------Connection Closed------')




class TinySoftData(Connection):
    
    def __init__(self):
        self.conn = pyTSL.Client("gtjazq", "gtja123", "tsl.tinysoft.com", 443)
        t = self.conn.login()
        if t == 1:
            print('------TSDB Connected, Good to go------')
        else:
            print('------Connection Failed-------')
        
    def close(self):
        
        self.conn.logout()
        print('------Connection Closed------')
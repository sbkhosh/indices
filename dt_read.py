#!/usr/bin/python3

import numpy as np
import os
import pandas as pd
import string
import yaml

from dt_help import Helper
from yahoofinancials import YahooFinancials

class DataProcessor():
    def __init__(self, input_directory, output_directory, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, output directory = {}, input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_prm_file))
        
    @Helper.timing
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)

    @Helper.timing
    def process(self):
        file_in = self.conf.get('file_in')
        use_cols = self.conf.get('use_cols')

        self.data = pd.read_excel('/'.join((self.input_directory,file_in)),sheet_name=[0,1,2,3,4],usecols=use_cols,header=None,skiprows=[0,1,2,3])
        
        # mapping columns numbers with the selected xlsx file letters 
        dct = dict(enumerate(string.ascii_uppercase))
        dct = {value:key for key, value in dct.items()}
        splits = use_cols.split(',')
        
        headers_num = []
        for el in splits:
            splt = el.split(':')
            lst = list(map(chr, range(ord(splt[0]),ord(splt[1])+1)))
            headers_num.append([dct[x] for x in lst])
            
        cols_daily = [ 'Open','High','Low','Close','Volume' ]
        cols_minute = [ 'Open','Close','High','Low','Volume' ]
        
        vix_index_daily = self.data[4][headers_num[0]].dropna()
        eu_index_daily = self.data[3][headers_num[0]].dropna()
        hk_index_daily = self.data[2][headers_num[0]].dropna()
        nikkei_index_daily = self.data[1][headers_num[0]].dropna()
        spmini_index_daily = self.data[0][headers_num[0]].dropna()
        
        vix_index_minute = self.data[4][headers_num[1]].dropna()
        eu_index_minute = self.data[3][headers_num[1]].dropna()
        hk_index_minute = self.data[2][headers_num[1]].dropna()
        nikkei_index_minute = self.data[1][headers_num[1]].dropna()
        spmini_index_minute = self.data[0][headers_num[1]].dropna()

        vix_index_daily = vix_index_daily.rename(columns=vix_index_daily.iloc[0]).drop(vix_index_daily.index[0]).set_index('Dates')
        eu_index_daily = eu_index_daily.rename(columns=eu_index_daily.iloc[0]).drop(eu_index_daily.index[0]).set_index('Dates')
        hk_index_daily = hk_index_daily.rename(columns=hk_index_daily.iloc[0]).drop(hk_index_daily.index[0]).set_index('Dates')
        nikkei_index_daily = nikkei_index_daily.rename(columns=nikkei_index_daily.iloc[0]).drop(nikkei_index_daily.index[0]).set_index('Dates')
        spmini_index_daily = spmini_index_daily.rename(columns=spmini_index_daily.iloc[0]).drop(spmini_index_daily.index[0]).set_index('Dates')

        vix_index_minute = vix_index_minute.rename(columns=vix_index_minute.iloc[0]).drop(vix_index_minute.index[0]).set_index('Dates')
        eu_index_minute = eu_index_minute.rename(columns=eu_index_minute.iloc[0]).drop(eu_index_minute.index[0]).set_index('Dates')
        hk_index_minute = hk_index_minute.rename(columns=hk_index_minute.iloc[0]).drop(hk_index_minute.index[0]).set_index('Dates')
        nikkei_index_minute = nikkei_index_minute.rename(columns=nikkei_index_minute.iloc[0]).drop(nikkei_index_minute.index[0]).set_index('Dates')
        spmini_index_minute = spmini_index_minute.rename(columns=spmini_index_minute.iloc[0]).drop(spmini_index_minute.index[0]).set_index('Dates')
        
        vix_index_daily.columns = cols_daily
        eu_index_daily.columns = cols_daily
        hk_index_daily.columns = cols_daily
        nikkei_index_daily.columns = cols_daily
        spmini_index_daily.columns = cols_daily
        
        vix_index_minute.columns = cols_minute
        eu_index_minute.columns = cols_minute
        hk_index_minute.columns = cols_minute
        nikkei_index_minute.columns = cols_minute
        spmini_index_minute.columns = cols_minute        

        self.vix_daily_dates = vix_index_daily.index
        self.eu_daily_dates = eu_index_daily.index
        self.hk_daily_dates = hk_index_daily.index
        self.nikkei_daily_dates = nikkei_index_daily.index
        self.spmini_daily_dates = spmini_index_daily.index

        self.vix_minute_dates = vix_index_minute.index
        self.eu_minute_dates = eu_index_minute.index
        self.hk_minute_dates = hk_index_minute.index
        self.nikkei_minute_dates = nikkei_index_minute.index
        self.spmini_minute_dates = spmini_index_minute.index
        
        # localized timezone to UTC
        vix_index_daily.index = vix_index_daily.index.tz_localize('UTC')
        eu_index_daily.index = eu_index_daily.index.tz_localize('UTC')
        hk_index_daily.index = hk_index_daily.index.tz_localize('UTC')
        nikkei_index_daily.index = nikkei_index_daily.index.tz_localize('UTC')
        spmini_index_daily.index = spmini_index_daily.index.tz_localize('UTC')

        vix_index_minute.index = vix_index_minute.index.tz_localize('UTC')
        eu_index_minute.index = eu_index_minute.index.tz_localize('UTC')
        hk_index_minute.index = hk_index_minute.index.tz_localize('UTC')
        nikkei_index_minute.index = nikkei_index_minute.index.tz_localize('UTC')
        spmini_index_minute.index = spmini_index_minute.index.tz_localize('UTC')
        
        vix_index_daily['H-L'] = vix_index_daily['High'] - vix_index_daily['Low']
        eu_index_daily['H-L'] = eu_index_daily['High'] - eu_index_daily['Low']
        hk_index_daily['H-L'] = hk_index_daily['High'] - hk_index_daily['Low']
        nikkei_index_daily['H-L'] = nikkei_index_daily['High'] - nikkei_index_daily['Low']
        spmini_index_daily['H-L'] = spmini_index_daily['High'] - spmini_index_daily['Low']

        vix_index_minute['H-L'] = vix_index_minute['High'] - vix_index_minute['Low']
        eu_index_minute['H-L'] = eu_index_minute['High'] - eu_index_minute['Low']
        hk_index_minute['H-L'] = hk_index_minute['High'] - hk_index_minute['Low']
        nikkei_index_minute['H-L'] = nikkei_index_minute['High'] - nikkei_index_minute['Low']
        spmini_index_minute['H-L'] = spmini_index_minute['High'] - spmini_index_minute['Low']
        
        vix_index_daily = vix_index_daily.astype(float)
        eu_index_daily = eu_index_daily.astype(float)
        hk_index_daily = hk_index_daily.astype(float)
        nikkei_index_daily = nikkei_index_daily.astype(float)
        spmini_index_daily = spmini_index_daily.astype(float)

        vix_index_minute = vix_index_minute.astype(float)
        eu_index_minute = eu_index_minute.astype(float)
        hk_index_minute = hk_index_minute.astype(float)
        nikkei_index_minute = nikkei_index_minute.astype(float)
        spmini_index_minute = spmini_index_minute.astype(float)
        
        vix_index_daily.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        eu_index_daily.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        hk_index_daily.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        nikkei_index_daily.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        spmini_index_daily.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)

        vix_index_minute.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        eu_index_minute.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        hk_index_minute.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        nikkei_index_minute.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        spmini_index_minute.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        
        self.vix_index_daily = vix_index_daily
        self.eu_index_daily = eu_index_daily
        self.hk_index_daily = hk_index_daily
        self.nikkei_index_daily = nikkei_index_daily
        self.spmini_index_daily = spmini_index_daily

        self.vix_index_minute = vix_index_minute
        self.eu_index_minute = eu_index_minute
        self.hk_index_minute = hk_index_minute
        self.nikkei_index_minute = nikkei_index_minute
        self.spmini_index_minute = spmini_index_minute
        
    def write_to(self,name,flag):
        filename = os.path.join(self.output_directory,name)
        try:
            if('csv' in flag):
                self.data.to_csv(str(name)+'.csv')
            elif('xls' in flag):
                self.data.to_excel(str(name)+'xls')
        except:
            raise ValueError("not supported format")


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

        self.data = pd.read_excel('/'.join((self.input_directory,file_in)),usecols=use_cols,header=None,skiprows=[0,1,2])
        
        # mapping columns numbers with the selected xlsx file letters 
        dct = dict(enumerate(string.ascii_uppercase))
        dct = {value:key for key, value in dct.items()}
        splits = use_cols.split(',')

        headers_num = []
        for el in splits:
            splt = el.split(':')
            lst = list(map(chr, range(ord(splt[0]),ord(splt[1])+1)))
            headers_num.append([dct[x] for x in lst])

        cols = [ 'Dates','Open','Close','High','Low','Volume' ]
            
        hk_index = self.data[headers_num[0]].dropna()
        nikkei_index = self.data[headers_num[1]].dropna()
        spmini_index = self.data[headers_num[2]].dropna()

        hk_index = hk_index.rename(columns=hk_index.iloc[0]).drop(hk_index.index[0]).set_index('Dates')
        nikkei_index = nikkei_index.rename(columns=nikkei_index.iloc[0]).drop(nikkei_index.index[0]).set_index('Dates')
        spmini_index = spmini_index.rename(columns=spmini_index.iloc[0]).drop(spmini_index.index[0]).set_index('Dates')

        # timezone of downloaded file is GMT-6 then convert to UTC
        hk_index.index = hk_index.index.tz_localize('Etc/GMT-6').tz_convert('UTC')
        nikkei_index.index = nikkei_index.index.tz_localize('Etc/GMT-6').tz_convert('UTC')
        spmini_index.index = spmini_index.index.tz_localize('Etc/GMT-6').tz_convert('UTC')

        hk_index['H-L'] = hk_index['High'] - hk_index['Low']
        nikkei_index['H-L'] = nikkei_index['High'] - nikkei_index['Low']
        spmini_index['H-L'] = spmini_index['High'] - spmini_index['Low']

        hk_index = hk_index.astype(float)
        nikkei_index = nikkei_index.astype(float)
        spmini_index = spmini_index.astype(float)
        
        hk_index.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        nikkei_index.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        spmini_index.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)

        self.hk_index = hk_index
        self.nikkei_index = nikkei_index
        self.spmini_index = spmini_index
                
    def write_to(self,name,flag):
        filename = os.path.join(self.output_directory,name)
        try:
            if('csv' in flag):
                self.data.to_csv(str(name)+'.csv')
            elif('xls' in flag):
                self.data.to_excel(str(name)+'xls')
        except:
            raise ValueError("not supported format")


#!/usr/bin/python3

import csv
import inspect
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import time
import yaml

from functools import wraps

class Helper():
    def __init__(self, input_directory, input_prm_file):
        self.input_directory = input_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, input parameter file  = {}'.format(self.input_directory, self.input_prm_file))

    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
            
    @staticmethod
    def timing(f):
        """Decorator for timing functions
        Usage:
        @timing
        def function(a):
        pass
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()
            print('function:%r took: %2.2f sec' % (f.__name__,  end - start))
            return(result)
        return wrapper

    @staticmethod
    def get_delim(filename):
        with open(filename, 'r') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
        return(dialect.delimiter)

    @staticmethod
    def get_class_membrs(clss):
        res = inspect.getmembers(clss, lambda a:not(inspect.isroutine(a)))
        return(res)

    @staticmethod
    def check_missing_data(data):
        print(data.isnull().sum().sort_values(ascending=False))

    @staticmethod
    def view_data(data):
        print(data.head())
        
    @staticmethod
    def adfuller_test(series):
        output = {}
        cmpr_t = lambda t, ctr: 'Stationary' if (t < ctr) else 'Non-stationary'
        cmpr_p = lambda p: 'Stationary' if (p < 0.05) else 'Non-stationary'
        
        r = sm.tsa.stattools.adfuller(series,autolag='AIC')
        output = {'ADF test_statistic': round(r[0],4), 'p-value': round(r[1],4), 'n_lags': round(r[2],4), 'n_obs': r[3],
                  'Critical value (1%)': round(r[4]['1%'],4),
                  'Critical value (5%)': round(r[4]['5%'],4),
                  'Critical value (10%)': round(r[4]['10%'],4),
                  'Stationary (1%)': cmpr_t(round(r[0],4),round(r[4]['1%'],4)),
                  'Stationary (5%)': cmpr_t(round(r[0],4),round(r[4]['5%'],4)),
                  'Stationary (10%)': cmpr_t(round(r[0],4),round(r[4]['10%'],4)),
                  'Stationary (p-value)': cmpr_p(round(r[1],4))}
        return(output)

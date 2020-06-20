#!/usr/bin/python3

import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

from datetime import datetime
from dt_help import Helper
from dt_read import DataProcessor
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None 
register_matplotlib_converters()

def get_data_all():
    obj_reader = DataProcessor('data_in','data_out','conf_model.yml')
    obj_reader.read_prm()
    obj_reader.process()
    return(obj_reader.hk_index_daily,obj_reader.nikkei_index_daily,obj_reader.spmini_index_daily,obj_reader.eu_index_daily,obj_reader.vix_index_daily, \
           obj_reader.hk_index_minute,obj_reader.nikkei_index_minute,obj_reader.spmini_index_minute,obj_reader.eu_index_minute,obj_reader.vix_index_minute, \
           obj_reader.hk_daily_dates,obj_reader.nikkei_daily_dates,obj_reader.spmini_daily_dates,obj_reader.eu_daily_dates,obj_reader.vix_daily_dates, \
           obj_reader.hk_minute_dates,obj_reader.nikkei_minute_dates,obj_reader.spmini_minute_dates,obj_reader.eu_minute_dates,obj_reader.vix_minute_dates)

def get_conf_helper():
    obj_helper = DataProcessor('data_in','data_out','conf_help.yml')
    obj_helper.read_prm()
    cut_cluster = obj_helper.conf.get('cut_cluster')
    cut_cluster_num = obj_helper.conf.get('cut_cluster_num')
    max_cluster_rep = obj_helper.conf.get('max_cluster_rep')
    return(cut_cluster,cut_cluster_num,max_cluster_rep)

def all_common_dates():
    start_date_minute = pd.to_datetime('2020-01-06 00:00:00').tz_localize('UTC')
    end_date_minute = pd.to_datetime('2020-06-01 23:59:59').tz_localize('UTC')
   
    # find daily dates that are common to all indices (filtering out holdidays included in one market but not the other for example)
    start_date_daily = pd.to_datetime('2020-'"{:02d}"'-'"{:02d}".format(start_date_minute.month,start_date_minute.day))
    end_date_daily = pd.to_datetime('2020-'"{:02d}"'-'"{:02d}".format(end_date_minute.month,end_date_minute.day))

    days_hk = pd.DataFrame(hk_daily_dates[hk_daily_dates >= start_date_daily])
    days_nk = pd.DataFrame(nikkei_daily_dates[nikkei_daily_dates >= start_date_daily])
    days_sp = pd.DataFrame(spmini_daily_dates[spmini_daily_dates >= start_date_daily])
    days_eu = pd.DataFrame(eu_daily_dates[eu_daily_dates >= start_date_daily])

    tmp_hk = [ pd.to_datetime(el) for el in days_hk['Dates'].values ]
    tmp_nk = [ pd.to_datetime(el) for el in days_nk['Dates'].values ]
    tmp_sp = [ pd.to_datetime(el) for el in days_sp['Dates'].values ]
    tmp_eu = [ pd.to_datetime(el) for el in days_eu['Dates'].values ]

    days_hk_filter = [ "{:02d}"'-'"{:02d}"'-'"{:02d}".format(el.year,el.month,el.day) for el in tmp_hk ]
    days_nk_filter = [ "{:02d}"'-'"{:02d}"'-'"{:02d}".format(el.year,el.month,el.day) for el in tmp_nk ]
    days_sp_filter = [ "{:02d}"'-'"{:02d}"'-'"{:02d}".format(el.year,el.month,el.day) for el in tmp_sp ]
    days_eu_filter = [ "{:02d}"'-'"{:02d}"'-'"{:02d}".format(el.year,el.month,el.day) for el in tmp_eu ]
    
    elements_in_all = list(set.intersection(*map(set, [days_hk_filter,days_nk_filter,days_sp_filter,days_eu_filter])))

    days_all = sorted(elements_in_all, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
    days_all = [pd.to_datetime(el) for el in days_all]
    return(days_all)

def get_acf_pacf(lst):
    lag_acf = acf(lst, nlags=300)    
    lag_pacf = pacf(lst, nlags=30) # , method='ols'

    fig = plt.figure(figsize=(32,20))

    ax1 = fig.add_subplot(211)
    ax1.plot(lag_acf,marker='+')
    ax1.axhline(y=0,linestyle='--',color='gray')
    ax1.axhline(y=-1.96/np.sqrt(len(lag_acf)),linestyle='--',color='gray')
    ax1.axhline(y=1.96/np.sqrt(len(lag_acf)),linestyle='--',color='gray')
    plt.title('ACF')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')
    plt.grid(True)
    plt.tight_layout()    

    ax2 = fig.add_subplot(212)
    ax2.plot(lag_pacf,marker='+')
    ax2.axhline(y=0,linestyle='--',color='blue')
    ax2.axhline(y=-1.96/np.sqrt(len(lag_pacf)),linestyle='--',color='blue')
    ax2.axhline(y=1.96/np.sqrt(len(lag_pacf)),linestyle='--',color='blue')
    plt.title('PACF')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')
    plt.grid(True)
    plt.tight_layout()    
    plt.show()

def auto_corr(ohlc):
    days_all = all_common_dates()
    
    df_hang_select_all = []
    df_nikkei_select_all = []
    df_spmini500_select_all = []
    df_eustoxx50_select_all = []
    
    # for each daily date take a full 24hr session (for all indices)
    for i,el in enumerate(days_all):
        sd = pd.to_datetime(str(el.date()) + ' 00:00:00').tz_localize('UTC')
        ed = pd.to_datetime(str(el.date()) + ' 23:59:59').tz_localize('UTC')
        
        mask_hang = (df_hk_minute.index >= sd) & (df_hk_minute.index <= ed)
        mask_nikkei = (df_nikkei_minute.index >= sd) & (df_nikkei_minute.index <= ed)
        mask_spmini500 = (df_spmini500_minute.index >= sd) & (df_spmini500_minute.index <= ed)
        mask_eustoxx50 = (df_eustoxx50_minute.index >= sd) & (df_eustoxx50_minute.index <= ed)

        df_hang_select = df_hk_minute.loc[mask_hang]
        df_nikkei_select = df_nikkei_minute.loc[mask_nikkei]
        df_spmini500_select = df_spmini500_minute.loc[mask_spmini500]
        df_eustoxx50_select = df_eustoxx50_minute.loc[mask_eustoxx50]

        df_hang_select['rate_ret'] = df_hang_select[ohlc].pct_change()
        df_nikkei_select['rate_ret'] = df_nikkei_select[ohlc].pct_change()
        df_spmini500_select['rate_ret'] = df_spmini500_select[ohlc].pct_change()
        df_eustoxx50_select['rate_ret'] = df_eustoxx50_select[ohlc].pct_change()

        df_hang_select.dropna(inplace=True)
        df_nikkei_select.dropna(inplace=True)
        df_spmini500_select.dropna(inplace=True)
        df_eustoxx50_select.dropna(inplace=True)

        df_hang_select_all.append(df_hang_select['rate_ret'].T)
        df_nikkei_select_all.append(df_nikkei_select['rate_ret'].T)
        df_spmini500_select_all.append(df_spmini500_select['rate_ret'].T)
        df_eustoxx50_select_all.append(df_eustoxx50_select['rate_ret'].T)

    lst = df_hang_select_all[0].values
    get_acf_pacf(lst)
    plot_acf(lst)
    plt.show()
    
    # df_hang_merge = pd.concat(df_hang_select_all,axis=1)
    # df_nikkei_merge = pd.concat(df_nikkei_select_all,axis=1)
    # df_spmini500_merge = pd.concat(df_spmini500_select_all,axis=1)
    # df_eustoxx50_merge = pd.concat(df_eustoxx50_select_all,axis=1)

    # df_hang_merge.fillna(0,inplace=True)
    # df_nikkei_merge.fillna(0,inplace=True)
    # df_spmini500_merge.fillna(0,inplace=True)
    # df_eustoxx50_merge.fillna(0,inplace=True)

    # df_hang_merge.columns = ["{:02d}"'-'"{:02d}".format(el.month,el.day)  for el in days_all ]
    # df_nikkei_merge.columns = ["{:02d}"'-'"{:02d}".format(el.month,el.day)  for el in days_all ]
    # df_spmini500_merge.columns = ["{:02d}"'-'"{:02d}".format(el.month,el.day)  for el in days_all ]
    # df_eustoxx50_merge.columns = ["{:02d}"'-'"{:02d}".format(el.month,el.day)  for el in days_all ]
    
    return(0)

def check_norm(df):
    df.dropna(inplace=True)
    fig = plt.figure(figsize=(32,20))
    ax1 = fig.add_subplot(121)
    ax1.hist(df['inst_freq'], bins=params['nbins'], label='inst_freq')
    plt.title('Distributions')

    ax2 = fig.add_subplot(122)  
    sm.qqplot(df['inst_freq'],ax=ax2,color='r',fit=True, line='45')
    plt.title('Distributions')
    plt.show()

    # shapiro test
    p_value = scipy.stats.shapiro(df['inst_freq'])[1]
    if p_value <= 0.05:
        print("Null hypothesis of normality is rejected.")
    else:
        print("Null hypothesis of normality is accepted.")
        
if __name__ == '__main__':
    obj_helper = Helper('data_in','conf_help.yml')
    obj_helper.read_prm()
    
    fontsize = obj_helper.conf['font_size']
    matplotlib.rcParams['axes.labelsize'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = fontsize
    matplotlib.rcParams['ytick.labelsize'] = fontsize
    matplotlib.rcParams['legend.fontsize'] = fontsize
    matplotlib.rcParams['axes.titlesize'] = fontsize
    matplotlib.rcParams['text.color'] = 'k'

    df_hk_daily, df_nikkei_daily, df_spmini500_daily, df_eustoxx50_daily, df_vix_daily, \
    df_hk_minute, df_nikkei_minute, df_spmini500_minute, df_eustoxx50_minute, df_vix_minute, \
    hk_daily_dates,nikkei_daily_dates,spmini_daily_dates,eu_daily_dates,vix_daily_dates, \
    hk_minute_dates,nikkei_minute_dates,spmini_minute_dates,eu_minute_dates,vix_minute_dates = get_data_all()

    auto_corr('Close')



    

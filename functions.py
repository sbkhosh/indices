#!/usr/bin/python3

import base64
import clusterlib
import dash_bootstrap_components as dbc
import datetime
import itertools
import json
import math
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np    
import os
import pandas as pd
import plotly.graph_objs as go
import scipy
import seaborn as sns
import re
import styles
import sys
import warnings
import yaml

from dt_read import DataProcessor
from dt_help import Helper
from pandas.plotting import register_matplotlib_converters
from plotly.figure_factory import create_2d_density
from PyEMD import EMD
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import pdist
from statsmodels.sandbox.regression.predstd import wls_prediction_std

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

def stats_dataframes_ohlc(ohlc,days_all):
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

        df_hang_select['norm'] = (df_hang_select[ohlc].values - np.mean(df_hang_select[ohlc].values)) / np.max(df_hang_select[ohlc].values)
        df_nikkei_select['norm'] = (df_nikkei_select[ohlc].values - np.mean(df_nikkei_select[ohlc].values)) / np.max(df_nikkei_select[ohlc].values)
        df_spmini500_select['norm'] = (df_spmini500_select[ohlc].values - np.mean(df_spmini500_select[ohlc].values)) / np.max(df_spmini500_select[ohlc].values)  
        df_eustoxx50_select['norm'] = (df_eustoxx50_select[ohlc].values - np.mean(df_eustoxx50_select[ohlc].values)) / np.max(df_eustoxx50_select[ohlc].values) 
        
        df_hang_select_all.append(df_hang_select)
        df_nikkei_select_all.append(df_nikkei_select)
        df_spmini500_select_all.append(df_spmini500_select)
        df_eustoxx50_select_all.append(df_eustoxx50_select)
        
    return(df_hang_select_all,df_nikkei_select_all,df_spmini500_select_all,df_eustoxx50_select_all)
    
def cluster_draw(df_all, method, metric, max_cluster, selected_cluster, ts_space=5):
    df_res, Z, ddata, dm = clusterlib.maxclust_draw_rep(df_all.iloc[:,:], method, metric, int(max_cluster), 5)

    filename_0 = 'data_out/max_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)
    image_name_0 = filename_0+".png"
    location_0 = os.getcwd() + '/' + image_name_0
    with open('%s' %location_0, "rb") as image_file_0:
        encoded_string_0 = base64.b64encode(image_file_0.read()).decode()
    encoded_image_0 = "data:image/png;base64," + encoded_string_0

    clusterlib.get_dtw_uniq_cluster(df_res, df_all, method, metric, max_cluster, selected_cluster)
    filename_1 = 'data_out/dtw_uniq_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(selected_cluster)
    image_name_1=filename_1+".png"
    location_1 = os.getcwd() + '/' + image_name_1
    with open('%s' %location_1, "rb") as image_file_1:
        encoded_string_1 = base64.b64encode(image_file_1.read()).decode()
    encoded_image_1 = "data:image/png;base64," + encoded_string_1

    return(encoded_image_0,df_res,encoded_image_1)

def plot_recplot(filename_1,filename_2,filename_3,filename_4):
    image_name_1 = filename_1+".png"
    location_1 = os.getcwd() + '/' + image_name_1
    with open('%s' %location_1, "rb") as image_file_1:
        encoded_string_1 = base64.b64encode(image_file_1.read()).decode()
    encoded_image_1 = "data:image/png;base64," + encoded_string_1

    image_name_2 = filename_2+".png"
    location_2 = os.getcwd() + '/' + image_name_2
    with open('%s' %location_2, "rb") as image_file_2:
        encoded_string_2 = base64.b64encode(image_file_2.read()).decode()
    encoded_image_2 = "data:image/png;base64," + encoded_string_2

    image_name_3 = filename_3+".png"
    location_3 = os.getcwd() + '/' + image_name_3
    with open('%s' %location_3, "rb") as image_file_3:
        encoded_string_3 = base64.b64encode(image_file_3.read()).decode()
    encoded_image_3 = "data:image/png;base64," + encoded_string_3

    image_name_4 = filename_4+".png"
    location_4 = os.getcwd() + '/' + image_name_4
    with open('%s' %location_4, "rb") as image_file_4:
        encoded_string_4 = base64.b64encode(image_file_4.read()).decode()
    encoded_image_4 = "data:image/png;base64," + encoded_string_4
    
    return(encoded_image_1,encoded_image_2,encoded_image_3,encoded_image_4)
    
#####################################################################################################################################################################################
    
def df_to_table(df):
    return(dbc.Table.from_dataframe(df,
                                    bordered=True,
                                    dark=False,
                                    hover=True,
                                    responsive=True,
                                    striped=True))

#####################################################################################################################################################################################
    
def mdd(df):
    df.dropna(inplace=True)
    hwm = [0]
    drawdown = pd.Series(index = df.index)
    duration = pd.Series(index = df.index)
    
    for t in range(1, len(df.index)):
        cur_hwm = max(hwm[t-1], df[[el for el in df.columns if 'rate_ret' in el]].iloc[t].values)
        hwm.append(cur_hwm)
        drawdown[t]= hwm[t] - df[[el for el in df.columns if 'rate_ret' in el]].iloc[t]
        duration[t]= 0 if drawdown[t] == 0 else duration[t-1] + 1

    mdd = drawdown.max()
    mdd_duration = duration.max()
    volatility = np.std(df[[el for el in df.columns if 'rate_ret' in el]].values)

    res1 = df[[el for el in df.columns if 'cum_ret' in el]].iloc[-1].values[0]
    res2 = df[[el for el in df.columns if 'cum_ret' in el]].iloc[0].values[0]
    growth = res1/res2 - 1.0
    return({'mdd': round(mdd * 100.0,3), 'volatility': round(volatility * 100.0,3), 'growth': round(growth * 100.0,3)})

#####################################################################################################################################################################################
    
def stats_ohlc(df_1,df_2,df_3,df_4,day_in,ohlcv):
    len_df = [ len(df_1), len(df_2), len(df_3), len(df_4) ]
    markets = ['Hang Seng', 'Nikkei225', 'eMiniSP500', 'EuroStoxx50']

    df_1[ohlcv + '_rate_ret'] = df_1[ohlcv].pct_change().dropna()
    df_1[ohlcv + '_rate_ret'].iloc[0] = 0.0
    df_1[ohlcv + '_cum_ret'] = (1.0+df_1[ohlcv + '_rate_ret']).cumprod()
    df_1[ohlcv + '_cum_ret'].iloc[0] = 1.0

    df_2[ohlcv + '_rate_ret'] = df_2[ohlcv].pct_change().dropna()
    df_2[ohlcv + '_rate_ret'].iloc[0] = 0.0
    df_2[ohlcv + '_cum_ret'] = (1.0+df_2[ohlcv + '_rate_ret']).cumprod()
    df_2[ohlcv + '_cum_ret'].iloc[0] = 1.0

    df_3[ohlcv + '_rate_ret'] = df_3[ohlcv].pct_change().dropna()
    df_3[ohlcv + '_rate_ret'].iloc[0] = 0.0
    df_3[ohlcv + '_cum_ret'] = (1.0+df_3[ohlcv + '_rate_ret']).cumprod()
    df_3[ohlcv + '_cum_ret'].iloc[0] = 1.0

    df_4[ohlcv + '_rate_ret'] = df_4[ohlcv].pct_change().dropna()
    df_4[ohlcv + '_rate_ret'].iloc[0] = 0.0
    df_4[ohlcv + '_cum_ret'] = (1.0+df_4[ohlcv + '_rate_ret']).cumprod()
    df_4[ohlcv + '_cum_ret'].iloc[0] = 1.0   
    
    df_res = pd.DataFrame(index=[0,1,2,3],columns=['day_sess','st_sess','ed_sess','mdd','volatility','growth','max_ret_time','min_ret_time',
                                                   'mean_return','pos_return','up_ratio','down_ratio','index_market'])

    for i,el in enumerate([df_1,df_2,df_3,df_4]):
        res = mdd(el)

        df_res['day_sess'].iloc[i] = pd.to_datetime(day_in).date()
        df_res['st_sess'].iloc[i] = "{:02d}"':'"{:02d}"':'"{:02d}".format(el.index[0].hour,el.index[0].minute,el.index[0].second)
        df_res['ed_sess'].iloc[i] = "{:02d}"':'"{:02d}"':'"{:02d}".format(el.index[-1].hour,el.index[-1].minute,el.index[-1].second)
        df_res['mdd'].iloc[i] = res['mdd']
        df_res['volatility'].iloc[i] = res['volatility']
        df_res['growth'].iloc[i] = res['growth']

        # index for the maximum return value
        idx_max_ret = el[[ohlcv + '_rate_ret']].idxmax()
        idx_min_ret = el[[ohlcv + '_rate_ret']].idxmin()
        df_res['max_ret_time'].iloc[i] = "{:02d}"':'"{:02d}"':'"{:02d}".format(pd.to_datetime(idx_max_ret.values[0]).hour,
                                                                        pd.to_datetime(idx_max_ret.values[0]).minute,
                                                                        pd.to_datetime(idx_max_ret.values[0]).second)
        df_res['min_ret_time'].iloc[i] = "{:02d}"':'"{:02d}"':'"{:02d}".format(pd.to_datetime(idx_min_ret.values[0]).hour,
                                                                        pd.to_datetime(idx_min_ret.values[0]).minute,
                                                                        pd.to_datetime(idx_min_ret.values[0]).second)
        df_res['mean_return'].iloc[i] = round(el[ohlcv + '_rate_ret'].mean() * 100,5)
        df_res['pos_return'].iloc[i] = res['growth'] > 0.0
        df_res['up_ratio'].iloc[i] = round(100.0 * len(list(filter(lambda x: (x < 0), np.diff(el[ohlcv + '_rate_ret'].values)))) / float(len_df[i] - 1),2)
        df_res['down_ratio'].iloc[i] = round(100.0 * len(list(filter(lambda x: (x >= 0), np.diff(el[ohlcv + '_rate_ret'].values)))) / float(len_df[i] - 1),2)
        df_res['index_market'].iloc[i] = markets[i]

    return(df_res)

#####################################################################################################################################################################################
    
def movingaverage(df, window_size='30T'):
    return(df.rolling(window=window_size).mean())

def movingstd(df, window_size='30T'):
    return(df.rolling(window=window_size).std())
    
def bbands(price, window_size='30T', num_of_std=2):
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std  = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return(rolling_mean, upper_band, lower_band)

#####################################################################################################################################################################################
    
def fig_raw_plot(df,freq,index_val,wnd):
    data = [dict(
        type = 'candlestick',
        open = df['Open'],
        high = df['High'],
        low = df['Low'],
        close = df['Close'],
        x = df.index,
        yaxis = 'y2',
        name = index_val,
        increasing = dict(line = dict( color = styles.INCREASING_COLOR)),
        decreasing = dict(line = dict( color = styles.DECREASING_COLOR)),
    )]

    layout = dict()
    fig = dict(data=data,layout=layout)
    
    fig['layout'] = dict()
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['xaxis'] = dict(rangeselector = dict( visible = True))
    fig['layout']['yaxis'] = dict(domain = [0, 0.2], showticklabels = False, autoscale=True)
    fig['layout']['yaxis2'] = dict(domain = [0.2, 0.8])
    fig['layout']['legend'] = dict(orientation = 'h', y=0.9, x=0.3, yanchor='bottom')
    fig['layout']['margin'] = dict(t=40, b=40, r=40, l=40)
    fig['layout']['width'] = 1800
    fig['layout']['height'] = 1200

    range_buttons_1 = [dict(count=1,
                            label='1mo',
                            step='month',
                            stepmode='backward'),
                       dict(count=10,
                            label='10day',
                            step='day',
                            stepmode='backward'),
                       dict(count=5,
                            label='5day',
                            step='day',
                            stepmode='backward'),
                       ]

    range_buttons_2 = [dict(count=24,
                            label='24hr',
                            step='hour',
                            stepmode='backward'),                      
                       dict(count=12,
                            label='12hr',
                            step='hour',
                            stepmode='backward'),                      
                       dict(count=1,
                            label='1hr',
                            step='hour',
                            stepmode='backward'),
                       dict(count=30,
                            label='30min',
                            step='minute',
                            stepmode='backward'),
                       dict(count=15,
                            label='15min',
                            step='minute',
                            stepmode='backward'),
                    ]

    if(freq == 'daily'):
        range_buttons_all = [dict(count=1,label='reset',step='all')] + range_buttons_1 + [ dict(step='all') ]
    elif(freq == '15min'):
        range_buttons_all = [dict(count=1,label='reset',step='all')] + range_buttons_1 + range_buttons_2 + [ dict(step='all') ]
        
    rangeselector=dict(
        visibe = True,
        x = 0, y = 0.9,
        bgcolor = 'rgba(150, 200, 250, 0.4)',
        font = dict( size = 16 ),
        buttons=list(range_buttons_all))

    fig['layout']['xaxis']['rangeselector'] = rangeselector

    mv_y = movingaverage(df['Close'],window_size=wnd)
    mv_x = list(df.index)

    fig['data'].append( dict( x=mv_x, y=mv_y, type='scatter', mode='lines', 
                            line = dict( width = 1 ),
                            marker = dict( color = styles.color_3 ),
                            yaxis = 'y2', name='Moving Average' ) )

    colors = []
    
    for i in range(len(df['Close'])):
        if i != 0:
            if df['Close'][i] > df['Close'][i-1]:
                colors.append(styles.INCREASING_COLOR)
            else:
                colors.append(styles.DECREASING_COLOR)
        else:
            colors.append(styles.DECREASING_COLOR)

    fig['data'].append(dict(x=df.index, y=df['Volume'],                         
                            marker=dict(color=colors),
                            type='bar', yaxis='y', name='Volume'))

    bb_avg, bb_upper, bb_lower = bbands(df['Close'],window_size=wnd)
    
    fig['data'].append( dict( x=df.index, y=bb_upper, type='scatter', yaxis='y2', 
                            line = dict( width = 1.5 ),
                            marker=dict(color=styles.color_2), hoverinfo='none', 
                            legendgroup='Bollinger Bands', name='Bollinger Bands') )

    fig['data'].append( dict( x=df.index, y=bb_lower, type='scatter', yaxis='y2',
                            line = dict( width = 1.5 ),
                            marker=dict(color=styles.color_2), hoverinfo='none',
                            legendgroup='Bollinger Bands', showlegend=False ) )
    return(fig)

#####################################################################################################################################################################################
    
def fig_scatter_plot(df,freq,index_val,wnd):
    data = [dict(
        type='scatter',
        mode='lines', 
        x = df.index,
        y = df['H-L'],
        yaxis = 'y2',
        name = index_val + '_H-L' ,
        line = dict(width=2),
        marker = dict(color=styles.color_4)
        )]

    layout = dict()
    fig = dict(data=data,layout=layout)
    
    fig['layout'] = dict()
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['xaxis'] = dict(rangeselector = dict( visible = True))
    fig['layout']['yaxis'] = dict(domain = [0, 0.2], showticklabels = False, autoscale=True)
    fig['layout']['yaxis2'] = dict(domain = [0.2, 0.8])
    fig['layout']['legend'] = dict(orientation = 'h', y=0.9, x=0.3, yanchor='bottom')
    fig['layout']['margin'] = dict(t=40, b=40, r=40, l=40)
    fig['layout']['width'] = 1800
    fig['layout']['height'] = 1200

    range_buttons_1 = [dict(count=1,
                            label='1mo',
                            step='month',
                            stepmode='backward'),
                       dict(count=10,
                            label='10day',
                            step='day',
                            stepmode='backward'),
                       dict(count=5,
                            label='5day',
                            step='day',
                            stepmode='backward'),
                       ]

    range_buttons_2 = [dict(count=24,
                            label='24hr',
                            step='hour',
                            stepmode='backward'),                      
                       dict(count=12,
                            label='12hr',
                            step='hour',
                            stepmode='backward'),                      
                       dict(count=1,
                            label='1hr',
                            step='hour',
                            stepmode='backward'),
                       dict(count=30,
                            label='30min',
                            step='minute',
                            stepmode='backward'),
                       dict(count=15,
                            label='15min',
                            step='minute',
                            stepmode='backward'),
                    ]

    if(freq == 'daily'):
        range_buttons_all = [dict(count=1,label='reset',step='all')] + range_buttons_1 + [ dict(step='all') ]
    elif(freq == '15min'):
        range_buttons_all = [dict(count=1,label='reset',step='all')] + range_buttons_1 + range_buttons_2 + [ dict(step='all') ]
        
    rangeselector=dict(
        visibe = True,
        x = 0, y = 0.9,
        bgcolor = 'rgba(150, 200, 250, 0.4)',
        font = dict( size = 16 ),
        buttons=list(range_buttons_all))

    fig['layout']['xaxis']['rangeselector'] = rangeselector

    mv_y = movingaverage(df['H-L'],window_size=wnd)
    mv_x = list(df.index)

    fig['data'].append(dict(x=mv_x, y=mv_y, type='scatter', mode='lines', 
                            line = dict( width = 1 ),
                            marker = dict( color = styles.color_3 ),
                            yaxis = 'y2', name='Moving Average'+'_H-L'))

    colors = []
    
    for i in range(len(df['H-L'])):
        if i != 0:
            if df['H-L'][i] > df['H-L'][i-1]:
                colors.append(styles.INCREASING_COLOR)
            else:
                colors.append(styles.DECREASING_COLOR)
        else:
            colors.append(styles.DECREASING_COLOR)

    fig['data'].append(dict(x=df.index, y=df['Volume'],                         
                            marker=dict(color=colors),
                            type='bar', yaxis='y', name='Volume'))

    bb_avg, bb_upper, bb_lower = bbands(df['H-L'],window_size=wnd)
    
    fig['data'].append( dict( x=df.index, y=bb_upper, type='scatter', yaxis='y2', 
                            line = dict( width = 1.5 ),
                            marker=dict(color=styles.color_2), hoverinfo='none', 
                            legendgroup='Bollinger Bands', name='Bollinger Bands') )

    fig['data'].append( dict( x=df.index, y=bb_lower, type='scatter', yaxis='y2',
                            line = dict( width = 1.5 ),
                            marker=dict(color=styles.color_2), hoverinfo='none',
                            legendgroup='Bollinger Bands', showlegend=False ) )
    return(fig)

#####################################################################################################################################################################################
    
def data_pairplot(df):
    filename = 'data_out/pairplot.png'

    if(os.path.exists(filename)):
        image_name=filename
        with open(image_name, 'rb') as f:
            encoded_string = base64.b64encode(f.read()).decode()
        encoded_image = "data:image/png;base64," + encoded_string
    else:
        fig, ax = plt.subplots(figsize=(32,16))
        g = sns.PairGrid(df)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(sns.kdeplot, n_levels=10)
        plt.tight_layout()
        plt.savefig(filename)

        image_name=filename
        with open(image_name, 'rb') as f:
            encoded_string = base64.b64encode(f.read()).decode()
        encoded_image = "data:image/png;base64," + encoded_string
    return(encoded_image)

#####################################################################################################################################################################################
    
def data_pairheat(df,title,thrs):
    corr_matrix = df.corr()
    corr_matrix_filtered = corr_matrix[(corr_matrix > thrs) & (corr_matrix < 1)]
    
    data = [dict(
        type = 'heatmap',
        z = corr_matrix_filtered.values,
        y = corr_matrix_filtered.columns,
        x = corr_matrix_filtered.index,
        name = title,
        annotation_text = corr_matrix_filtered.round(2).values,
        xaxis = 'x',
        yaxis = 'y',
        )]

    layout = dict()
    fig = dict(data=data,layout=layout)

    fig['layout'] = dict()
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['tickangle'] = 45
    fig['layout']['xaxis'] = dict(zeroline=False,showticklabels=True)
    fig['layout']['yaxis'] = dict(zeroline=False,showticklabels=True)
    fig['layout']['legend'] = dict(orientation = 'h', y=0.9, x=0.3, yanchor='right')
    fig['layout']['margin'] = dict(t=40, b=40, r=40, l=100)
    fig['layout']['width'] = 1800
    fig['layout']['height'] = 1200
    fig['layout']['autosize'] = True
    fig['layout']['colorscale'] = [[0, 'rgba(255, 255, 255,0)'], [1, '#a3a7b0']]
    fig['layout']['thicknessmode'] = 'pixels'
    fig['layout']['thickness'] = 50
    fig['layout']['lenmode'] = 'pixels'
    fig['layout']['len'] = 1000

    return(fig,corr_matrix_filtered)

#####################################################################################################################################################################################
    
def data_heatmap(df):
    data = [dict(
        type = 'heatmap',
        z = df.values,
        x = df.columns.values,
        y = df.index,
        name = 'All indices',
        )]

    layout = dict()
    fig = dict(data=data,layout=layout)
    
    fig['layout'] = dict()
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['tickangle'] = -45
    fig['layout']['xaxis'] = dict(zeroline=True,showticklabels=True)
    fig['layout']['yaxis'] = dict(zeroline=True,showticklabels=True)
    fig['layout']['legend'] = dict(orientation = 'h', y=0.9, x=0.3, yanchor='right')
    fig['layout']['margin'] = dict(t=40, b=40, r=40, l=100)
    fig['layout']['width'] = 1200
    fig['layout']['height'] = 1200
    fig['layout']['autosize'] = True
    fig['layout']['xaxis_tickformat'] = '%Y-%m-%d %H:%M:%S'
    fig['layout']['xaxis_ntick'] = len(df.index)
    fig['layout']['xgap'] = 3
    fig['layout']['ygap'] = 3
    fig['layout']['colorscale'] = 'rgba(255, 255, 255,0)'
    fig['layout']['thicknessmode'] = 'pixels'
    fig['layout']['thickness'] = 50
    fig['layout']['lenmode'] = 'pixels'
    fig['layout']['len'] =200    

    return(fig)

#####################################################################################################################################################################################
    
def candlestick_plot(df,freq,index_val):
    data = [dict(
        type = 'candlestick',
        x = df.index,
        open = df['Open'],
        high = df['High'],
        low = df['Low'],
        close = df['Close'],
        yaxis = 'y2',
        name = index_val,
        increasing = dict(line = dict( color = styles.INCREASING_COLOR)),
        decreasing = dict(line = dict( color = styles.DECREASING_COLOR)),
        )]

    layout = dict()
    fig = dict(data=data,layout=layout)
    
    fig['layout'] = dict()
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['xaxis'] = dict(rangeselector = dict( visible = True))
    fig['layout']['yaxis'] = dict(domain = [0, 0.2], showticklabels = False, autoscale=True)
    fig['layout']['yaxis2'] = dict(domain = [0.2, 0.8])
    fig['layout']['legend'] = dict(orientation = 'h', y=0.9, x=0.3, yanchor='bottom')
    fig['layout']['margin'] = dict(t=40, b=40, r=40, l=40)
    fig['layout']['width'] = 1800
    fig['layout']['height'] = 1200

    range_buttons_1 = [dict(count=1,
                            label='1mo',
                            step='month',
                            stepmode='backward'),
                       dict(count=10,
                            label='10day',
                            step='day',
                            stepmode='backward'),
                       dict(count=5,
                            label='5day',
                            step='day',
                            stepmode='backward'),
                       ]

    range_buttons_2 = [dict(count=24,
                            label='24hr',
                            step='hour',
                            stepmode='backward'),                      
                       dict(count=12,
                            label='12hr',
                            step='hour',
                            stepmode='backward'),                      
                       dict(count=1,
                            label='1hr',
                            step='hour',
                            stepmode='backward'),
                       dict(count=30,
                            label='30min',
                            step='minute',
                            stepmode='backward'),
                       dict(count=15,
                            label='15min',
                            step='minute',
                            stepmode='backward'),
                    ]

    if(freq == 'daily'):
        range_buttons_all = [dict(count=1,label='reset',step='all')] + range_buttons_1 + [ dict(step='all') ]
    elif(freq == '15min'):
        range_buttons_all = [dict(count=1,label='reset',step='all')] + range_buttons_1 + range_buttons_2 + [ dict(step='all') ]
        
    rangeselector=dict(
        visibe = True,
        x = 0, y = 0.9,
        bgcolor = 'rgba(150, 200, 250, 0.4)',
        font = dict( size = 16 ),
        buttons=list(range_buttons_all))

    fig['layout']['xaxis']['rangeselector'] = rangeselector

    colors = []
    
    for i in range(len(df['Close'])):
        if i != 0:
            if df['Close'][i] > df['Close'][i-1]:
                colors.append(styles.INCREASING_COLOR)
            else:
                colors.append(styles.DECREASING_COLOR)
        else:
            colors.append(styles.DECREASING_COLOR)

    fig['data'].append(dict(x=df.index, y=df['Volume'],                         
                            marker=dict(color=colors),
                            type='bar', yaxis='y', name='Volume'))
    return(fig)

#####################################################################################################################################################################################
    
def fig_dist_comp(index_val_1,index_val_2,df_hk_1,df_hk_2,df_nk_1,df_nk_2,df_sp_1,df_sp_2,df_eu_1,df_eu_2,df_vix_1,df_vix_2,ohlc_1,ohlc_2):
    names = list(itertools.product(['Nikkei225','HangSeng','eMiniSP500','EuroStoxx50','VIX'],repeat = 2))
    df_names = [(df_nk_1, df_nk_2), (df_nk_1, df_hk_2), (df_nk_1, df_sp_2), (df_nk_1, df_eu_2), (df_nk_1, df_vix_2),
                (df_hk_1, df_nk_2), (df_hk_1, df_hk_2), (df_hk_1, df_sp_2), (df_hk_1, df_eu_2), (df_hk_1, df_vix_2),
                (df_sp_1, df_nk_2), (df_sp_1, df_hk_2), (df_sp_1, df_sp_2), (df_sp_1, df_eu_2), (df_sp_1, df_vix_2),
                (df_eu_1, df_nk_2), (df_eu_1, df_hk_2), (df_eu_1, df_sp_2), (df_eu_1, df_eu_2), (df_eu_1, df_vix_2),
                (df_vix_1, df_nk_2), (df_vix_1, df_hk_2), (df_vix_1, df_sp_2), (df_vix_1, df_eu_2), (df_vix_1, df_vix_2)]

    dct = dict(zip(names,df_names))
    matched = [el[1] for el in dct.items() if((index_val_1,index_val_2)==el[0])][0]
    
    xx_ret = matched[0][ohlc_1].pct_change().dropna()
    yy_ret = matched[1][ohlc_2].pct_change().dropna()
    xx_log_ret = (np.log(matched[0][ohlc_1]) - np.log(matched[0][ohlc_1].shift(1))).dropna()
    yy_log_ret = (np.log(matched[1][ohlc_2]) - np.log(matched[1][ohlc_2].shift(1))).dropna()
    
    data_1 = [dict(
        type = 'histogram2dcontour',
        x = xx_ret,
        y = yy_ret,
        name = index_val_1 + 'vs. ' + index_val_2,
        coloscale = 'Blues',
        xaxis = 'x',
        yaxis = 'y'
        )]
        
    layout = dict()
    fig_1 = dict(data=data_1,layout=layout)

    fig_1['layout'] = dict()
    fig_1['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig_1['layout']['xaxis'] = dict(title=index_val_1 + '-' + ohlc_1)
    fig_1['layout']['yaxis'] = dict(title=index_val_2 + '-' + ohlc_2)
    fig_1['layout']['width'] = 800
    fig_1['layout']['height'] = 800

    fig_1['data'].append(dict(x=xx_ret,
                              y=yy_ret,
                              xaxis = 'x',
                              yaxis = 'y',
                              type='scatter',
                              mode='markers',
                              line = dict( width = 1 ),
                              marker = dict(
                                  color = 'rgba(0,0,0,0.3)',
                                  size = 3)))

    fig_1['data'].append(dict(y=yy_ret,
                            xaxis = 'x2',
                            type='histogram',
                            line = dict( width = 1 ),
                            marker = dict(
                                color = 'rgba(0,0,0,0.2)',
                                size = 3),
                            title = dict(title=index_val_1 + '-' + ohlc_1)))

    fig_1['data'].append(dict(x=xx_ret,
                            yaxis = 'y2',
                            type='histogram',
                            line = dict( width = 1 ),
                            marker = dict(
                                color = 'rgba(0,0,0,0.2)',
                                size = 3),
                            title = dict(title=index_val_1 + '-' + ohlc_1)))

    fig_1['layout']['autosize'] = False
    fig_1['layout']['xaxis'] = dict(zeroline = False,domain = [0,0.85],showgrid = False)
    fig_1['layout']['yaxis'] = dict(zeroline = False,domain = [0,0.85],showgrid = False)
    fig_1['layout']['xaxis2'] = dict(zeroline = False,domain = [0.85,1],showgrid = False)
    fig_1['layout']['yaxis2'] = dict(zeroline = False,domain = [0.85,1],showgrid = False)
    fig_1['layout']['bargap'] = 0
    fig_1['layout']['hovermode'] = 'closest'
    fig_1['layout']['showlegend'] = False

    data_2 = [dict(
        type = 'histogram2dcontour',
        x = xx_log_ret,
        y = yy_log_ret,
        name = index_val_1 + 'vs. ' + index_val_2,
        coloscale = 'Blues',
        xaxis = 'x',
        yaxis = 'y'
        )]
        
    layout = dict()
    fig_2 = dict(data=data_2,layout=layout)

    fig_2['layout'] = dict()
    fig_2['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig_2['layout']['xaxis'] = dict(title=index_val_1 + '-' + ohlc_1)
    fig_2['layout']['yaxis'] = dict(title=index_val_2 + '-' + ohlc_2)
    fig_2['layout']['width'] = 800
    fig_2['layout']['height'] = 800

    fig_2['data'].append(dict(x=xx_log_ret,
                            y=yy_log_ret,
                            xaxis = 'x',
                            yaxis = 'y',
                            type='scatter',
                            mode='markers',
                            line = dict( width = 1 ),
                            marker = dict(
                                color = 'rgba(0,0,0,0.3)',
                                size = 3)))

    fig_2['data'].append(dict(y=yy_log_ret,
                            xaxis = 'x2',
                            type='histogram',
                            line = dict( width = 1 ),
                            marker = dict(
                                color = 'rgba(0,0,0,0.2)',
                                size = 3),
                            title = dict(title=index_val_1 + '-' + ohlc_1)))

    fig_2['data'].append(dict(x=xx_log_ret,
                            yaxis = 'y2',
                            type='histogram',
                            line = dict( width = 1 ),
                            marker = dict(
                                color = 'rgba(0,0,0,0.2)',
                                size = 3),
                            title = dict(title=index_val_1 + '-' + ohlc_1)))

    fig_2['layout']['autosize'] = False
    fig_2['layout']['xaxis'] = dict(zeroline = False,domain = [0,0.85],showgrid = False)
    fig_2['layout']['yaxis'] = dict(zeroline = False,domain = [0,0.85],showgrid = False)
    fig_2['layout']['xaxis2'] = dict(zeroline = False,domain = [0.85,1],showgrid = False)
    fig_2['layout']['yaxis2'] = dict(zeroline = False,domain = [0.85,1],showgrid = False)
    fig_2['layout']['bargap'] = 0
    fig_2['layout']['hovermode'] = 'closest'
    fig_2['layout']['showlegend'] = False

    return(fig_1,fig_2)

#####################################################################################################################################################################################
    
def fig_lag(df_hk,df_nk,df_sp,df_eu,ohlc,hours_diff):
    df_all_last_hang,df_all_last_nikkei,df_all_last_spmini500,df_all_last_eustoxx50,df_all_first_hang,df_all_first_nikkei,df_all_first_spmini500,df_all_first_eustoxx50 = \
      get_all_first_last(df_hk,df_nk,df_sp,df_eu,ohlc,hours_diff)

    # dataframe with all data
    pairs_all = ['Hang_Seng_last','Nikkei225_last','eMiniSP500_last','EuroStoxx50_last',
                 'Hang_Seng_first','Nikkei225_first','eMiniSP500_first','EuroStoxx50_first']

    df_all_first_last = pd.concat([df_all_last_hang,df_all_last_nikkei,df_all_last_spmini500,df_all_last_eustoxx50,
                                   df_all_first_hang,df_all_first_nikkei,df_all_first_spmini500,df_all_first_eustoxx50],axis=1)
    df_all_first_last.columns = pairs_all
    
    pairs_comb = [ el[0]+'__'+el[1] for el in list(itertools.combinations(pairs_all,2)) ]
    pairs_dfs = [(df_all_last_hang, df_all_last_nikkei), (df_all_last_hang, df_all_last_spmini500), (df_all_last_hang, df_all_last_eustoxx50), (df_all_last_hang, df_all_first_hang),
                 (df_all_last_hang, df_all_first_nikkei), (df_all_last_hang, df_all_first_spmini500), (df_all_last_hang, df_all_first_eustoxx50),(df_all_last_nikkei, df_all_last_spmini500),
                 (df_all_last_nikkei, df_all_last_eustoxx50), (df_all_last_nikkei, df_all_first_hang), (df_all_last_nikkei, df_all_first_nikkei), (df_all_last_nikkei, df_all_first_spmini500),
                 (df_all_last_nikkei, df_all_first_eustoxx50), (df_all_last_spmini500, df_all_last_eustoxx50), (df_all_last_spmini500, df_all_first_hang), (df_all_last_spmini500, df_all_first_nikkei),
                 (df_all_last_spmini500, df_all_first_spmini500), (df_all_last_spmini500, df_all_first_eustoxx50),(df_all_last_eustoxx50, df_all_first_hang), (df_all_last_eustoxx50, df_all_first_nikkei),
                 (df_all_last_eustoxx50, df_all_first_spmini500), (df_all_last_eustoxx50, df_all_first_eustoxx50), (df_all_first_hang, df_all_first_nikkei), (df_all_first_hang, df_all_first_spmini500),
                 (df_all_first_hang, df_all_first_eustoxx50), (df_all_first_nikkei, df_all_first_spmini500), (df_all_first_nikkei, df_all_first_eustoxx50), (df_all_first_spmini500, df_all_first_eustoxx50)]

    dct_pairs = dict(zip(pairs_comb,pairs_dfs))

    df_summary = pd.DataFrame(index=pairs_comb,columns=['same direction','opposite direction'])
    for i,(k,v) in enumerate(dct_pairs.items()):
        res = np.sign(v[0]['growth'].multiply(v[1]['growth'],axis='index'))
        df_summary.iloc[i] = [len(res[res==1]),len(res[res==-1])]

    df_summary['pair_1'] = [el.split('__')[0] for el in df_summary.index]
    df_summary['pair_2'] = [el.split('__')[1] for el in df_summary.index]

    lag_table = df_to_table(df_summary[::-1])
    
    # plot results
    data_1_last = [dict(
        type = 'bar',
        x = df_all_last_hang.index,
        y = df_all_last_hang['growth'],
        name = 'Hang Seng - Last',
    )]

    layout = dict()
    fig = dict(data=data_1_last,layout=layout)
    
    fig['layout'] = dict()
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['xaxis'] = {'automargin': True}
    fig['layout']['yaxis'] = dict(automargin=True,title='cumulative return')
    fig['layout']['width'] = 1800
    fig['layout']['height'] = 1200

    data_2_last = dict(
        type = 'bar',
        x = df_all_last_nikkei.index,
        y = df_all_last_nikkei['growth'],
        name = 'Nikkei225 - Last',
    )

    data_3_last = dict(
        type = 'bar',
        x = df_all_last_spmini500.index,
        y = df_all_last_spmini500['growth'],
        name = 'eMini SP500 - Last',
    )

    data_4_last = dict(
        type = 'bar',
        x = df_all_last_eustoxx50.index,
        y = df_all_last_eustoxx50['growth'],
        name = 'Eurostoxx50 - Last',
    )

    # first
    data_1_first = dict(
        type = 'bar',
        x = df_all_first_hang.index,
        y = df_all_first_hang['growth'],
        name = 'Hang Seng - First',
    )
    
    data_2_first = dict(
        type = 'bar',
        x = df_all_first_nikkei.index,
        y = df_all_first_nikkei['growth'],
        name = 'Nikkei225 - First',
    )

    data_3_first = dict(
        type = 'bar',
        x = df_all_first_spmini500.index,
        y = df_all_first_spmini500['growth'],
        name = 'eMini SP500 - First',
    )

    data_4_first = dict(
        type = 'bar',
        x = df_all_first_eustoxx50.index,
        y = df_all_first_eustoxx50['growth'],
        name = 'Eurostoxx50 - First',
    )

    fig['data'].append(data_2_last)
    fig['data'].append(data_3_last)
    fig['data'].append(data_4_last)
    fig['data'].append(data_1_first)
    fig['data'].append(data_2_first)
    fig['data'].append(data_3_first)
    fig['data'].append(data_4_first)

    return(fig,lag_table)

def fig_adf(df_1,df_2,df_3,df_4):
    data_1 = [dict(
        type = 'bar',
        x = df_1['Dates'],
        y = df_1['Stationary (p-value)'],
        name = 'Hang Seng',
    )]

    layout_1 = dict()
    fig_1 = dict(data=data_1,layout=layout_1)
    
    fig_1['layout'] = dict()
    fig_1['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig_1['layout']['xaxis'] = {'automargin': True}
    fig_1['layout']['yaxis'] = dict(automargin=True,title='')
    fig_1['layout']['width'] = 1800
    fig_1['layout']['height'] = 300

    data_2 = [dict(
        type = 'bar',
        x = df_2['Dates'],
        y = df_2['Stationary (p-value)'],
        name = 'Nikkei225',
    )]

    layout_2 = dict()
    fig_2 = dict(data=data_2,layout=layout_2)
    
    fig_2['layout'] = dict()
    fig_2['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig_2['layout']['xaxis'] = {'automargin': True}
    fig_2['layout']['yaxis'] = dict(automargin=True,title='')
    fig_2['layout']['width'] = 1800
    fig_2['layout']['height'] = 300
    
    data_3 = [dict(
        type = 'bar',
        x = df_3['Dates'],
        y = df_3['Stationary (p-value)'],
        name = 'eMini SP500',
    )]

    layout_3 = dict()
    fig_3 = dict(data=data_3,layout=layout_3)
    
    fig_3['layout'] = dict()
    fig_3['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig_3['layout']['xaxis'] = {'automargin': True}
    fig_3['layout']['yaxis'] = dict(automargin=True,title='')
    fig_3['layout']['width'] = 1800
    fig_3['layout']['height'] = 300
    
    data_4 = [dict(
        type = 'bar',
        x = df_4['Dates'],
        y = df_4['Stationary (p-value)'],
        name = 'Eurostoxx50',
    )]

    layout_4 = dict()
    fig_4 = dict(data=data_4,layout=layout_4)
    
    fig_4['layout'] = dict()
    fig_4['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig_4['layout']['xaxis'] = {'automargin': True}
    fig_4['layout']['yaxis'] = dict(automargin=True,title='')
    fig_4['layout']['width'] = 1800
    fig_4['layout']['height'] = 300
    
    return(fig_1,fig_2,fig_3,fig_4)

def fig_sampen(df_1,df_2,df_3,df_4):
    data_1 = [dict(
        type = 'bar',
        x = df_1['Dates'],
        y = df_1['SampEn'],
        name = 'Hang Seng',
    )]

    layout_1 = dict()
    fig_1 = dict(data=data_1,layout=layout_1)
    
    fig_1['layout'] = dict()
    fig_1['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig_1['layout']['xaxis'] = {'automargin': True}
    fig_1['layout']['yaxis'] = dict(automargin=True,title='')
    fig_1['layout']['width'] = 1800
    fig_1['layout']['height'] = 300

    data_2 = [dict(
        type = 'bar',
        x = df_2['Dates'],
        y = df_2['SampEn'],
        name = 'Nikkei225',
    )]

    layout_2 = dict()
    fig_2 = dict(data=data_2,layout=layout_2)
    
    fig_2['layout'] = dict()
    fig_2['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig_2['layout']['xaxis'] = {'automargin': True}
    fig_2['layout']['yaxis'] = dict(automargin=True,title='')
    fig_2['layout']['width'] = 1800
    fig_2['layout']['height'] = 300
    
    data_3 = [dict(
        type = 'bar',
        x = df_3['Dates'],
        y = df_3['SampEn'],
        name = 'eMini SP500',
    )]

    layout_3 = dict()
    fig_3 = dict(data=data_3,layout=layout_3)
    
    fig_3['layout'] = dict()
    fig_3['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig_3['layout']['xaxis'] = {'automargin': True}
    fig_3['layout']['yaxis'] = dict(automargin=True,title='')
    fig_3['layout']['width'] = 1800
    fig_3['layout']['height'] = 300
    
    data_4 = [dict(
        type = 'bar',
        x = df_4['Dates'],
        y = df_4['SampEn'],
        name = 'Eurostoxx50',
    )]

    layout_4 = dict()
    fig_4 = dict(data=data_4,layout=layout_4)
    
    fig_4['layout'] = dict()
    fig_4['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig_4['layout']['xaxis'] = {'automargin': True}
    fig_4['layout']['yaxis'] = dict(automargin=True,title='')
    fig_4['layout']['width'] = 1800
    fig_4['layout']['height'] = 300
    
    return(fig_1,fig_2,fig_3,fig_4)

    

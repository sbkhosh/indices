#!/usr/bin/python3

import colorlover as cl
import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import datetime
import itertools
import math
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np    
import os
import pandas as pd
import plotly.graph_objs as go
import scipy
import scipy.cluster.hierarchy as hac
import statsmodels.api as sm
import seaborn as sns
import sys
import warnings
import yaml

from dash.dependencies import Output, Input, State
from datetime import datetime
from dt_read import DataProcessor
from pandas.plotting import register_matplotlib_converters
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import pdist
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from plotly.figure_factory import create_2d_density


warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None 
register_matplotlib_converters()
   
color_1 = '#c0b8c3'
color_2 = '#3e75cf'
color_3 = '#42483f'
color_4 = '#7b1ec3'
color_5 = '#450eae'
color_6 = 'black'

INCREASING_COLOR = 'green'
DECREASING_COLOR = 'red'

STYLE_1 = {'font-family': 'Calibri','font-size':50,'width': '33%','display':'inline-block'}
STYLE_2 = {'font-family': 'Calibri','font-size':18,'width': '50%','display':'inline-block'}
STYLE_3 = {'width': '50%', 'float': 'left', 'display': 'inline-block'}
STYLE_4 = {'height': '100%', 'width': '100%', 'float': 'left', 'padding': 90}
STYLE_5 = {'font-family': 'Calibri', 'color': color_1}
STYLE_6 = {'color': color_5}
STYLE_7 = {'height': '50%', 'width': '100%'}
STYLE_8 = {'font-family': 'Calibri', 'color': color_6}
STYLE_9 = {'width': '33%', 'float': 'left', 'display': 'inline-block'}
STYLE_10 = {'width': '16.6%', 'float': 'left', 'display': 'inline-block'}
STYLE_11 = {'height': '200%', 'width': '200%', 'float': 'left', 'padding': 90}

def get_data_all():
    obj_reader = DataProcessor('data_in','data_out','conf_model.yml')
    obj_reader.read_prm()
    obj_reader.process()
    return(obj_reader.hk_index_daily,obj_reader.nikkei_index_daily,obj_reader.spmini_index_daily,obj_reader.eu_index_daily,obj_reader.vix_index_daily, \
           obj_reader.hk_index_minute,obj_reader.nikkei_index_minute,obj_reader.spmini_index_minute,obj_reader.eu_index_minute,obj_reader.vix_index_minute, \
           obj_reader.hk_daily_dates,obj_reader.nikkei_daily_dates,obj_reader.spmini_daily_dates,obj_reader.eu_daily_dates,obj_reader.vix_daily_dates, \
           obj_reader.hk_minute_dates,obj_reader.nikkei_minute_dates,obj_reader.spmini_minute_dates,obj_reader.eu_minute_dates,obj_reader.vix_minute_dates)

df_hk_daily, df_nikkei_daily, df_spmini500_daily, df_eustoxx50_daily, df_vix_daily, \
df_hk_minute, df_nikkei_minute, df_spmini500_minute, df_eustoxx50_minute, df_vix_minute, \
hk_daily_dates,nikkei_daily_dates,spmini_daily_dates,eu_daily_dates,vix_daily_dates, \
hk_minute_dates,nikkei_minute_dates,spmini_minute_dates,eu_minute_dates,vix_minute_dates = get_data_all()

def get_df_choice(freq,index_val,wnd):
    if(freq == 'daily' and index_val=='HangSeng'):
        df = df_hk_daily
    elif(freq == 'daily' and index_val == 'Nikkei225'):
        df = df_nikkei_daily
    elif(freq == 'daily' and index_val == 'eMiniSP500'):
        df = df_spmini500_daily
    elif(freq == 'daily' and index_val == 'EuroStoxx50'):
        df = df_eustoxx50_daily
    elif(freq == 'daily' and index_val == 'VIX'):
        df = df_vix_daily
    elif(freq == '15min' and index_val == 'HangSeng'):
        df = df_hk_minute
    elif(freq == '15min' and index_val == 'Nikkei225'):
        df = df_nikkei_minute
    elif(freq == '15min' and index_val == 'eMiniSP500'):
        df = df_spmini500_minute
    elif(freq == '15min' and index_val == 'EuroStoxx50'):
        df = df_eustoxx50_minute
    elif(freq == '15min' and index_val == 'VIX'):
        df = df_vix_minute
    return(df)

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
    return({'mdd': mdd * 100.0, 'volatility': volatility * 100.0, 'growth': growth * 100.0})
    
def stats_ohlcv(df_1,df_2,df_3,df_4,day_in,ohlcv):
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
    
    df_res = pd.DataFrame(index=[0,1,2,3],columns=['st_session','ed_session','mdd','volatility','growth','max_ret_time','min_ret_time',
                                                   'mean_return','std_return','pos_return_session','up_mvs_ratio','down_mvs_ratio','market','day_session'])

    for i,el in enumerate([df_1,df_2,df_3,df_4]):
        res = mdd(el)

        df_res['st_session'].iloc[i] = "{:02d}"':'"{:02d}"':'"{:02d}".format(el.index[0].hour,el.index[0].minute,el.index[0].second)
        df_res['ed_session'].iloc[i] = "{:02d}"':'"{:02d}"':'"{:02d}".format(el.index[-1].hour,el.index[-1].minute,el.index[-1].second)
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
        df_res['mean_return'].iloc[i] = el[ohlcv + '_rate_ret'].mean()
        df_res['std_return'].iloc[i] = el[ohlcv + '_rate_ret'].std()
        df_res['pos_return_session'].iloc[i] = res['growth'] > 0.0
        df_res['up_mvs_ratio'].iloc[i] = 100.0 * len(list(filter(lambda x: (x < 0), np.diff(el[ohlcv + '_rate_ret'].values)))) / float(len_df[i] - 1)
        df_res['down_mvs_ratio'].iloc[i] = 100.0 * len(list(filter(lambda x: (x >= 0), np.diff(el[ohlcv + '_rate_ret'].values)))) / float(len_df[i] - 1)
        df_res['market'].iloc[i] = markets[i]
        df_res['day_session'].iloc[i] = pd.to_datetime(day_in).date()

    return(df_res)
    
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
        increasing = dict(line = dict( color = INCREASING_COLOR)),
        decreasing = dict(line = dict( color = DECREASING_COLOR)),
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

    # Clip the ends
    # mv_x = mv_x[5:-5]
    # mv_y = mv_y[5:-5]

    fig['data'].append( dict( x=mv_x, y=mv_y, type='scatter', mode='lines', 
                            line = dict( width = 1 ),
                            marker = dict( color = color_3 ),
                            yaxis = 'y2', name='Moving Average' ) )

    colors = []
    
    for i in range(len(df['Close'])):
        if i != 0:
            if df['Close'][i] > df['Close'][i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)

    fig['data'].append(dict(x=df.index, y=df['Volume'],                         
                            marker=dict(color=colors),
                            type='bar', yaxis='y', name='Volume'))

    bb_avg, bb_upper, bb_lower = bbands(df['Close'],window_size=wnd)
    
    fig['data'].append( dict( x=df.index, y=bb_upper, type='scatter', yaxis='y2', 
                            line = dict( width = 1.5 ),
                            marker=dict(color=color_2), hoverinfo='none', 
                            legendgroup='Bollinger Bands', name='Bollinger Bands') )

    fig['data'].append( dict( x=df.index, y=bb_lower, type='scatter', yaxis='y2',
                            line = dict( width = 1.5 ),
                            marker=dict(color=color_2), hoverinfo='none',
                            legendgroup='Bollinger Bands', showlegend=False ) )
    return(fig)

def fig_scatter_plot(df,freq,index_val,wnd):
    data = [dict(
        type='scatter',
        mode='lines', 
        x = df.index,
        y = df['H-L'],
        yaxis = 'y2',
        name = index_val + '_H-L' ,
        line = dict(width=2),
        marker = dict(color=color_4)
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

    # Clip the ends
    # mv_x = mv_x[5:-5]
    # mv_y = mv_y[5:-5]

    fig['data'].append(dict(x=mv_x, y=mv_y, type='scatter', mode='lines', 
                            line = dict( width = 1 ),
                            marker = dict( color = color_3 ),
                            yaxis = 'y2', name='Moving Average'+'_H-L'))

    colors = []
    
    for i in range(len(df['H-L'])):
        if i != 0:
            if df['H-L'][i] > df['H-L'][i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)

    fig['data'].append(dict(x=df.index, y=df['Volume'],                         
                            marker=dict(color=colors),
                            type='bar', yaxis='y', name='Volume'))

    bb_avg, bb_upper, bb_lower = bbands(df['H-L'],window_size=wnd)
    
    fig['data'].append( dict( x=df.index, y=bb_upper, type='scatter', yaxis='y2', 
                            line = dict( width = 1.5 ),
                            marker=dict(color=color_2), hoverinfo='none', 
                            legendgroup='Bollinger Bands', name='Bollinger Bands') )

    fig['data'].append( dict( x=df.index, y=bb_lower, type='scatter', yaxis='y2',
                            line = dict( width = 1.5 ),
                            marker=dict(color=color_2), hoverinfo='none',
                            legendgroup='Bollinger Bands', showlegend=False ) )
    return(fig)

def fig_bar_plot(df,freq,index_val,wnd):
    data = [dict(
        type='bar',
        x = df.index,
        y = df['H-L'],
        yaxis = 'y2',
        name = index_val + '_H-L',
        line = dict(width=2),
        marker = dict(color=color_4)
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

    mv_y = movingstd(df['H-L'],window_size=wnd)
    mv_x = list(df.index)

    # Clip the ends
    # mv_x = mv_x[5:-5]
    # mv_y = mv_y[5:-5]

    fig['data'].append(dict(x=mv_x, y=mv_y, type='scatter', mode='lines', 
                            line = dict( width = 2 ),
                            marker = dict( color = color_3 ),
                            yaxis = 'y2', name='Moving Standard Deviation'+'_H-L'))

    colors = []
    
    for i in range(len(df['H-L'])):
        if i != 0:
            if df['H-L'][i] > df['H-L'][i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)

    fig['data'].append(dict(x=df.index, y=df['Volume'],                         
                            marker=dict(color=colors),
                            type='bar', yaxis='y', name='Volume'))


    if(freq == 'daily'):
        mask_hk = (df_hk_daily.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_hk_daily.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_hk_select = df_hk_daily.loc[mask_hk]
        fig_hk = candlestick_plot(df_hk_select,freq,'HangSeng')

        mask_nikkei = (df_nikkei_daily.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_nikkei_daily.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_nikkei_select = df_nikkei_daily.loc[mask_nikkei]
        fig_nikkei = candlestick_plot(df_nikkei_select,freq,'Nikkei225')

        mask_spmini500 = (df_spmini500_daily.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_spmini500_daily.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_spmini500_select = df_spmini500_daily.loc[mask_spmini500]
        fig_spmini500 = candlestick_plot(df_spmini500_select,freq,'eMiniSP500')

        mask_eustoxx50 = (df_eustoxx50_daily.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_eustoxx50_daily.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_eustoxx50_select = df_eustoxx50_daily.loc[mask_eustoxx50]
        fig_eustoxx50 = candlestick_plot(df_eustoxx50_select,freq,'EuroStoxx50')

        mask_vix = (df_vix_daily.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_vix_daily.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_vix_select = df_vix_daily.loc[mask_vix]
        fig_vix = candlestick_plot(df_vix_select,freq,'VIX')
    elif(freq == '15min'):
        mask_hk = (df_hk_minute.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_hk_minute.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_hk_select = df_hk_minute.loc[mask_hk]
        fig_hk = candlestick_plot(df_hk_select,freq,'HangSeng')

        mask_nikkei = (df_nikkei_minute.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_nikkei_minute.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_nikkei_select = df_nikkei_minute.loc[mask_nikkei]
        fig_nikkei = candlestick_plot(df_nikkei_select,freq,'Nikkei225')

        mask_spmini500 = (df_spmini500_minute.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_spmini500_minute.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_spmini500_select = df_spmini500_minute.loc[mask_spmini500]
        fig_spmini500 = candlestick_plot(df_spmini500_select,freq,'eMiniSP500')

        mask_eustoxx50 = (df_eustoxx50_minute.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_eustoxx50_minute.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_eustoxx50_select = df_eustoxx50_minute.loc[mask_eustoxx50]
        fig_eustoxx50 = candlestick_plot(df_eustoxx50_select,freq,'EuroStoxx50')
        
        mask_vix = (df_vix_minute.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_vix_minute.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_vix_select = df_vix_minute.loc[mask_vix]
        fig_vix = candlestick_plot(df_vix_select,freq,'VIX')
        
    return(fig,fig_hk,fig_nikkei,fig_spmini500,fig_eustoxx50,fig_vix)
    
def data_pairplot(df):
    df = df.astype(float)
    filename = 'data_out/pairplot.png'

    if(os.path.exists(filename)):
        image_name=filename
        with open(image_name, "rb") as f:
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
        with open(image_name, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        encoded_image = "data:image/png;base64," + encoded_string
    return(encoded_image)
    
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
   
def data_heatmap(df):
    data = [dict(
        type = 'heatmap',
        z = df.values,
        x = df.columns.values,
        y = df.index,
        name = 'All indices - Week 10th and 17th May',
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
        increasing = dict(line = dict( color = INCREASING_COLOR)),
        decreasing = dict(line = dict( color = DECREASING_COLOR)),
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
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)

    fig['data'].append(dict(x=df.index, y=df['Volume'],                         
                            marker=dict(color=colors),
                            type='bar', yaxis='y', name='Volume'))
    return(fig)

def fig_dist_comp(index_val_1,index_val_2,df_hk_1,df_hk_2,df_nk_1,df_nk_2,df_sp_1,df_sp_2,df_eu_1,df_eu_2,df_vix_1,df_vix_2,ohlc_1,ohlc_2):
    names = list(itertools.product(['Nikkei225','HangSeng','eMiniSP500','EuroStoxx50','VIX'],repeat = 2))
    df_names = [(df_nk_1, df_nk_2), (df_nk_1, df_hk_2), (df_nk_1, df_sp_2), (df_nk_1, df_eu_2), (df_nk_1, df_vix_2),
                (df_hk_1, df_nk_2), (df_hk_1, df_hk_2), (df_hk_1, df_sp_2), (df_hk_1, df_eu_2), (df_hk_1, df_vix_2),
                (df_sp_1, df_nk_2), (df_sp_1, df_hk_2), (df_sp_1, df_sp_2), (df_sp_1, df_eu_2), (df_sp_1, df_vix_2),
                (df_eu_1, df_nk_2), (df_eu_1, df_hk_2), (df_eu_1, df_sp_2), (df_eu_1, df_eu_2), (df_eu_1, df_vix_2),
                (df_vix_1, df_nk_2), (df_vix_1, df_hk_2), (df_vix_1, df_sp_2), (df_vix_1, df_eu_2), (df_vix_1, df_vix_2)]

    dct = dict(zip(names,df_names))
    matched = [el[1] for el in dct.items() if((index_val_1,index_val_2)==el[0])][0]
    
    print(index_val_1,index_val_2)
    print(matched[0],matched[1])

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

def table_stats_ohlcv(df_hk,df_nk,df_sp,df_eu,ohlc):
    start_date_minute = pd.to_datetime('2020-01-06 00:00:00').tz_localize('UTC')
    end_date_minute = pd.to_datetime('2020-06-01 23:59:00').tz_localize('UTC')
   
    # find daily dates that are common to all indices (filtering out holdidays included in one market but not the other for example)
    start_date_daily = pd.to_datetime('2020-'"{:02d}"'-'"{:02d}".format(start_date_minute.month,start_date_minute.day))
    end_date_daily = pd.to_datetime('2020-'"{:02d}"'-'"{:02d}".format(end_date_minute.month,end_date_minute.day))

    days_hk = pd.DataFrame(hk_daily_dates[hk_daily_dates >= start_date_daily])
    days_nk = pd.DataFrame(nikkei_daily_dates[nikkei_daily_dates >= start_date_daily])
    days_sp = pd.DataFrame(spmini_daily_dates[spmini_daily_dates >= start_date_daily])
    days_eu = pd.DataFrame(eu_daily_dates[eu_daily_dates >= start_date_daily])

    tmp_hk = [ pd.to_datetime(el) for el in days_hk['Dates'].values ]
    days_hk_filter = [ "{:02d}"'-'"{:02d}"'-'"{:02d}".format(el.year,el.month,el.day) for el in tmp_hk ]

    tmp_nk = [ pd.to_datetime(el) for el in days_nk['Dates'].values ]
    days_nk_filter = [ "{:02d}"'-'"{:02d}"'-'"{:02d}".format(el.year,el.month,el.day) for el in tmp_nk ]

    tmp_sp = [ pd.to_datetime(el) for el in days_sp['Dates'].values ]
    days_sp_filter = [ "{:02d}"'-'"{:02d}"'-'"{:02d}".format(el.year,el.month,el.day) for el in tmp_sp ]

    tmp_eu = [ pd.to_datetime(el) for el in days_eu['Dates'].values ]
    days_eu_filter = [ "{:02d}"'-'"{:02d}"'-'"{:02d}".format(el.year,el.month,el.day) for el in tmp_eu ]
    
    
    elements_in_all = list(set.intersection(*map(set, [days_hk_filter,days_nk_filter,days_sp_filter,days_eu_filter])))

    days_all = sorted(elements_in_all, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
    days_all = [pd.to_datetime(el) for el in days_all]
    
    # for each daily date take a full 24hr session (for all indices)
    stats_all = []
    for el in days_all:
        sd = pd.to_datetime(str(el.date()) + ' 00:00:00').tz_localize('UTC')
        ed = pd.to_datetime(str(el.date()) + ' 23:59:59').tz_localize('UTC')
        
        mask_hk = (df_hk.index >= sd) & (df_hk.index <= ed)
        mask_nk = (df_nk.index >= sd) & (df_nk.index <= ed)
        mask_sp = (df_sp.index >= sd) & (df_sp.index <= ed)
        mask_eu = (df_eu.index >= sd) & (df_eu.index <= ed)

        df_hk_select = df_hk.loc[mask_hk]
        df_nk_select = df_nk.loc[mask_nk]
        df_sp_select = df_sp.loc[mask_sp]
        df_eu_select = df_eu.loc[mask_eu]

        stats_all.append(stats_ohlcv(df_hk_select,df_nk_select,df_sp_select,df_eu_select,el,ohlc))

    df_all = pd.concat(stats_all,axis=0).groupby(['day_session', 'market']).agg(lambda x: x)
    print(df_all)
        
def nav_menu():
    nav = dbc.Nav(
        [
            dbc.NavLink("Raw plots", href='/page-1', id='page-1-link', style=STYLE_1),
            dbc.NavLink("Volume", href='/page-2', id='page-2-link', style=STYLE_1),
            dbc.NavLink("H-L & Volume", href='/page-3', id='page-3-link', style=STYLE_1),
            dbc.NavLink("H-L & Volatility", href='/page-4', id='page-4-link', style=STYLE_1),
            dbc.NavLink("Correlations", href='/page-5', id='page-5-link', style=STYLE_1),
            dbc.NavLink("Distribution of returns", href='/page-6', id='page-6-link', style=STYLE_1),
            dbc.NavLink("Statistics - OHLCV", href='/page-7', id='page-7-link', style=STYLE_1)
        ],
        pills=True
        )
    return(nav)
    
def df_to_table(df):
    return(dbc.Table.from_dataframe(df,
                                    bordered=True,
                                    dark=False,
                                    hover=True,
                                    responsive=True,
                                    striped=True))

##########################################################################################################################################################################################
#                                                                                        layout_1
##########################################################################################################################################################################################
def get_layout_1():
    html_res = \
    html.Div([
              html.Div([
                        html.Div(html.P([html.Br(),html.H5(html.B('For each chosen index, different metrics are shown. The moving average represented in the graph is based on the Close value. It is also possible to show for different backward windows. We notice that the most active sessions start from Mid-March for Nikkei225 and SP500 contracts and from beginning of May for Hang Seng')),html.Br(),html.H5(html.B('For Nikkei, the daily trend in terms of volume activity shows, during each cycle, the same pattern: a peak and gradual decrease')),html.Br(),html.H5(html.B('For Hang Seng, the daily trend in terms of volume activity shows a parabolic evolution during each cycle')),html.Br(),html.H5(html.B('For SP500, the daily trend in terms of volume activity shows a net uniform trend during half of a 24hr cycle and lesser activity during the other half'))]), style=STYLE_8),
                        html.Div([
                                  html.Div(html.H4('Frequency'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ohlc-dd-freq',
                                      options=[{'label': i, 'value': i} for i in ['daily','15min']],
                                      value='15min',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_9),
                        html.Div([
                                  html.Div(html.H4('Index choice'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ohlc-dd-index',
                                      options=[{'label': i, 'value': i} for i in ['Nikkei225','HangSeng','eMiniSP500','EuroStoxx50','VIX']],
                                      value='Nikkei225',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_9),
                        html.Div([
                                  html.Div(html.H4('Moving average time window'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ohlc-dd-mvavg',
                                      options=[{'label': str(i//60)+'hr', 'value': str(i)+'T'} for i in [60,120,240,720,1440]],
                                      value='60T',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_9),
                        html.Div([
                                  dcc.Graph(
                                      id = 'ohlc',
                                      style=STYLE_4)
                                      ])
                                      ])
              ])
    return(html_res)

##########################################################################################################################################################################################
#                                                                                        layout_2
##########################################################################################################################################################################################  
def get_layout_2():
    html_res = \
    html.Div([
              html.Div([
                        html.Div(html.P([html.Br(),html.H5(html.B('The methodology is to first cut out dates for which volume is insignificant by filtering on a specific date for each of the index. This is done by comparing the volume from the raw data plots')),html.Br(),html.H5(html.B('HangSeng: cut-off date = 2020-04-26')),html.Br(),html.H5(html.B('Nikkei225: cut-off date = 2020-03-08')),html.Br(),html.H5(html.B('eMiniSP500: cut-off date = 2020-03-08')),html.Br(),html.H5(html.B('After these cut-off dates the volumes have already spiked and got to tradable levels')),html.Br(),html.H5(html.B('Then we select the intersection of the previous cut-off dates, i.e. 2020-04-26'))]), style=STYLE_8)
                        ]),
              html.Div([
                        html.Div([
                                  html.Div(html.H4('Frequency'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ohlc-dd-freq-cut',
                                      options=[{'label': i, 'value': i} for i in ['daily','15min']],
                                      value='daily',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_9),
                        html.Div([
                                  html.Div(html.H4('Index choice'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ohlc-dd-index-cut',
                                      options=[{'label': i, 'value': i} for i in ['Nikkei225','HangSeng','eMiniSP500','EuroStoxx50','VIX']],
                                      value='Nikkei225',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_9),
                        html.Div([
                                  html.Div(html.H4('Moving average time window'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ohlc-dd-mvavg-cut',
                                      options=[{'label': str(i//60)+'hr', 'value': str(i)+'T'} for i in [60,120,240,720,1440]],
                                      value='60T',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_9),
                        html.Div([
                                  dcc.Graph(
                                      id = 'ohlc-cut',
                                      style=STYLE_4)
                                      ])
                                      ])
              ])
    return(html_res)

##########################################################################################################################################################################################
#                                                                                        layout_3
##########################################################################################################################################################################################  
def get_layout_3():
    html_res = \
    html.Div([
              html.Div([
                        html.Div(html.P([html.Br(),html.H5(html.B('Representation of High-Low prices focuses from the latest cut-off date of 2020-04-26 (for Hang Seng)')),html.Br(),html.H5(html.B('The Volume representation is relative to the movement of the High-Low price difference'))]), style=STYLE_8)
                        ]),
              html.Div([
                        html.Div([
                                  html.Div(html.H4('Frequency'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='hl-dd-freq',
                                      options=[{'label': i, 'value': i} for i in ['daily','15min']],
                                      value='daily',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_9),
                        html.Div([
                                  html.Div(html.H4('Index choice'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='hl-dd-index',
                                      options=[{'label': i, 'value': i} for i in ['Nikkei225','HangSeng','eMiniSP500','EuroStoxx50','VIX']],
                                      value='Nikkei225',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_9),
                        html.Div([
                                  html.Div(html.H4('Moving average time window'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='hl-dd-mvavg',
                                      options=[{'label': str(i//60)+'hr', 'value': str(i)+'T'} for i in [60,120,240,720,1440]],
                                      value='60T',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_9),
                        html.Div([
                                  dcc.Graph(
                                      id = 'hl',
                                      style=STYLE_4)
                                      ])
                                      ])
              ])
    return(html_res)
    
##########################################################################################################################################################################################
#                                                                                        layout_4
##########################################################################################################################################################################################
def get_layout_4():
    html_res = \
    html.Div([
              html.Div([
                        html.Div(html.P([html.Br(),html.H5(html.B('As described in previous tabs, 2020-04-26 is the cut-off dates for which common trading sessions exist between the 3 indices considered here')),html.Br(),html.H5(html.B('As the week commencing 3rd May contains reduced trading days for Japan (due to holidays), we focus on two full weeks starting 10th May and 17th May')),html.Br(),html.H5(html.B('In addition these time windows are far enough from the volatile periods of the beginning of the year. This enables to look into a more general patterns/relationship between the different indices'))]), style=STYLE_8)
                        ]),
              html.Div([
                        html.Div([
                                  html.Div(html.H4('Frequency'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='vol-dd-freq',
                                      options=[{'label': i, 'value': i} for i in ['daily','15min']],
                                      value='daily',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_9),
                        html.Div([
                                  html.Div(html.H4('Index choice'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='vol-dd-index',
                                      options=[{'label': i, 'value': i} for i in ['Nikkei225','HangSeng','eMiniSP500','EuroStoxx50','VIX']],
                                      value='Nikkei225',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_9),
                        html.Div([
                                  html.Div(html.H4('Moving Standard Deviation time window'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='vol-dd-mvstd',
                                      options=[{'label': str(i//60)+'hr', 'value': str(i)+'T'} for i in [60,120,240,720,1440]],
                                      value='60T',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_9),
                        html.Div([
                                  dcc.Graph(
                                      id = 'vol',
                                      style=STYLE_4)
                                      ]),
                        html.Div([
                                  html.Div(html.P([html.Br(),html.H2(html.B('Hang Seng Index (week 10th and 17th May)')),html.Br()]), style=STYLE_8),
                                  dcc.Graph(
                                      id = 'vol-focus-hk',
                                      style=STYLE_4)
                                      ]),
                        html.Div([
                                  html.Div(html.P([html.Br(),html.H2(html.B('Nikkei Index (week 10th and 17th May)')),html.Br()]), style=STYLE_8),
                                  dcc.Graph(
                                      id = 'vol-focus-nikkei',
                                      style=STYLE_4)
                                      ]),
                        html.Div([
                                  html.Div(html.P([html.Br(),html.H2(html.B('eMiniSP500 (week 10th and 17th May)')),html.Br()]), style=STYLE_8),
                                  dcc.Graph(
                                      id = 'vol-focus-spmini500',
                                      style=STYLE_4)
                                      ]),
                        html.Div([
                                  html.Div(html.P([html.Br(),html.H2(html.B('EuroStoxx50 (week 10th and 17th May)')),html.Br()]), style=STYLE_8),
                                  dcc.Graph(
                                      id = 'vol-focus-eurostoxx50',
                                      style=STYLE_4)
                                      ]),
                        html.Div([
                                  html.Div(html.P([html.Br(),html.H2(html.B('VIX (week 10th and 17th May)')),html.Br()]), style=STYLE_8),
                                  dcc.Graph(
                                      id = 'vol-focus-vix',
                                      style=STYLE_4)
                                      ]),
                        ])
              ])
    return(html_res)

##########################################################################################################################################################################################
#                                                                                        layout_5
##########################################################################################################################################################################################
def get_layout_5():
    html_res = \
    html.Div([
              html.Div([
                        html.Div(html.P([html.Br(),html.H5(html.B('Now we look at the corrleation between the OHLCV componenent of each index, focusing on the week of 10th and 17th of May')),html.Br(),html.H5(html.B('A pairplot give a better description on the relationship between the OHLCV components of the indices')),html.Br(),html.H5(html.B('These price components (OHLC) are then selected based on the chosen correlation threshold. Then the returns are computed for this time period'))]), style=STYLE_8)]),
              html.Div([
                        html.Div([
                                  html.Div(html.H4('Frequency'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='htpp-dd-freq',
                                      options=[{'label': i, 'value': i} for i in ['daily','15min']],
                                      value='daily',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
                        html.Div([
                                  html.Div(html.H4('threshold correlation'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='htpp-dd-thrs',
                                      options=[{'label': i, 'value': i} for i in [round(el,2) for el in np.linspace(-1.0,1.0,21)]],
                                      value=0.9,
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
                        html.Div([
                                  dcc.Graph(
                                      id = 'corr',
                                      style=STYLE_4)
                                      ]),
                        html.Div([
                                  html.Div(
                                      id='corr-table',
                                      className='tableDiv'
                                      )
                                  ],style=STYLE_4),
                        ]),
              ])
    return(html_res)

##########################################################################################################################################################################################
#                                                                                        layout_6
##########################################################################################################################################################################################  
def get_layout_6():
    html_res = \
    html.Div([
              html.Div([
                        html.Div(html.P([html.Br(),html.H5(html.B('For each index, a date can be chosen for which the distributions for the 15 minute data')),html.Br(),html.H5(html.B(''))]), style=STYLE_8)
                        ]),
              html.Div([
                        html.Div([
                                  html.Div(html.H4('Index Choice - 1'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ret-dd-idx-1',
                                      options=[{'label': i, 'value': i} for i in ['Nikkei225','HangSeng','eMiniSP500','EuroStoxx50','VIX']],
                                      value='Nikkei225',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
                        html.Div([
                                  html.Div(html.H4('Index Choice - 2'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ret-dd-idx-2',
                                      options=[{'label': i, 'value': i} for i in ['Nikkei225','HangSeng','eMiniSP500','EuroStoxx50','VIX']],
                                      value='eMiniSP500',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
                     ]),
              html.Div([
                        html.Div([
                                  html.Div(html.H4('Day - 1'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ret-dd-day-1',
                                      options=[{'label': i, 'value': int(i)} for i in range(1,31)],
                                      value=5,
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_10),
                        html.Div([
                                  html.Div(html.H4('Month - 1'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ret-dd-month-1',
                                      options=[{'label': i, 'value': int(i)} for i in range(1,7)],
                                      value=3,
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_10),
                        html.Div([
                                  html.Div(html.H4('OHLC Choice - 1'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ret-dd-ohlc-1',
                                      options=[{'label': i, 'value': i} for i in ['Open','High','Low','Close']],
                                      value='High',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_10),                        
                        html.Div([
                                  html.Div(html.H4('Day - 2'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ret-dd-day-2',
                                      options=[{'label': i, 'value': int(i)} for i in range(1,31)],
                                      value=5,
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_10),
                        html.Div([
                                  html.Div(html.H4('Month - 2'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ret-dd-month-2',
                                      options=[{'label': i, 'value': int(i)} for i in range(1,7)],
                                      value=3,
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_10),
                        html.Div([
                                  html.Div(html.H4('OHLC Choice - 2'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ret-dd-ohlc-2',
                                      options=[{'label': i, 'value': i} for i in ['Open','High','Low','Close']],
                                      value='High',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_10),
                        html.Div([
                                  html.Div(html.P([html.Br(),html.H2(html.B('Simple return between indices at different dates')),html.Br()]), style=STYLE_8),
                                  dcc.Graph(
                                      id = 'ret-dist-comp',
                                      style=STYLE_4
                                      )
                                      ],style=STYLE_3),
                        html.Div([
                                  html.Div(html.P([html.Br(),html.H2(html.B('Log-return between indices at different dates')),html.Br()]), style=STYLE_8),
                                  dcc.Graph(
                                      id = 'log-ret-dist-comp',
                                      style=STYLE_4
                                      )
                                      ],style=STYLE_3),
                         ])
              ])
    
    return(html_res)

##########################################################################################################################################################################################
#                                                                                        layout_7
##########################################################################################################################################################################################  
def get_layout_7():
    html_res = \
    html.Div([
              html.Div([
                        html.Div(html.P([html.Br(),html.H5(html.B('')),html.Br(),html.H5(html.B(''))]), style=STYLE_8)
                        ]),
              html.Div([
                        html.Div([
                                  html.Div(html.H4('OHLCV Choice'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='stats-ohlcv',
                                      options=[{'label': i, 'value': i} for i in ['Open','High','Low','Close','Volume']],
                                      value='High',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
                        html.Div([
                                  html.Div(html.P([html.Br(),html.H2(html.B('Summary statistics table')),html.Br()]), style=STYLE_8),
                                  html.Div(
                                      id='stats-table-1',
                                      className='tableDiv'
                                      )
                                  ],style=STYLE_4),
                         ])
              ])
    return(html_res)
    
###################
# core of the app #  
###################
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP],meta_tags=[{"content": "width=device-width"}])
app.config.suppress_callback_exceptions = True
app.layout = html.Div([
    html.Div([
        html.H1(nav_menu())]),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),    
    ],                     
)

@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1,8)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        return True, False, False, False, False, False, False
    return [pathname == f"/page-{i}" for i in range(1,8)]

##########################################################################################################################################################################################
#                                                                                        page_1
##########################################################################################################################################################################################
@app.callback(
    Output('ohlc-dd-mvavg', 'options'),
    [Input('ohlc-dd-freq', 'value')]
)
def update_mvavg_dropdown_1(freq):
    if(freq == 'daily'):
        options=[{'label': str(i)+'D', 'value': int(i)} for i in [15,30,60,90]]
    elif(freq == '15min'):
        options=[{'label': str(i//60)+'hr', 'value': str(i)+'T'} for i in [60,120,240,720,1440]]
    return(options)
   
page_1_layout = html.Div([ get_layout_1() ])

@app.callback(Output('ohlc', 'figure'),
              [Input('ohlc-dd-freq', 'value'),
               Input('ohlc-dd-index', 'value'),
               Input('ohlc-dd-mvavg', 'value'),],
)
def update_fig_1(freq,index_val,wnd):
    df = get_df_choice(freq,index_val,wnd)

    fig = fig_raw_plot(df,freq,index_val,wnd)
    return(fig)

##########################################################################################################################################################################################
#                                                                                        page_2
##########################################################################################################################################################################################
@app.callback(
    Output('ohlc-dd-mvavg-cut', 'options'),
    [Input('ohlc-dd-freq-cut', 'value')]
)
def update_mvavg_dropdown_2(freq):
    if(freq == 'daily'):
        options=[{'label': str(i)+'D', 'value': int(i)} for i in [15,30,60,90]]
    elif(freq == '15min'):
        options=[{'label': str(i//60)+'hr', 'value': str(i)+'T'} for i in [60,120,240,720,1440]]
    return(options)
    
page_2_layout = html.Div([ get_layout_2() ])

@app.callback(Output('ohlc-cut', 'figure'),
              [Input('ohlc-dd-freq-cut', 'value'),
               Input('ohlc-dd-index-cut', 'value'),
               Input('ohlc-dd-mvavg-cut', 'value'),],
)
def update_fig_2(freq,index_val,wnd):
    df = get_df_choice(freq,index_val,wnd)
    mask = (df.index >= pd.to_datetime('2020-03-01 00:00:00').tz_localize('UTC'))
        
    fig = fig_raw_plot(df.loc[mask],freq,index_val,wnd)
    return(fig)
    
##########################################################################################################################################################################################
#                                                                                        page_3
##########################################################################################################################################################################################
@app.callback(
    Output('hl-dd-mvavg', 'options'),
    [Input('hl-dd-freq', 'value')]
)
def update_mvavg_dropdown_3(freq):
    if(freq == 'daily'):
        options=[{'label': str(i)+'D', 'value': int(i)} for i in [15,30,60,90]]
    elif(freq == '15min'):
        options=[{'label': str(i//60)+'hr', 'value': str(i)+'T'} for i in [60,120,240,720,1440]]
    return(options)
    
page_3_layout = html.Div([ get_layout_3() ])

@app.callback(Output('hl', 'figure'),
              [Input('hl-dd-freq', 'value'),
               Input('hl-dd-index', 'value'),
               Input('hl-dd-mvavg', 'value'),],
)              
def update_fig_3(freq,index_val,wnd):
    df = get_df_choice(freq,index_val,wnd)
    mask = (df.index >= pd.to_datetime('2020-03-01 00:00:00').tz_localize('UTC'))
        
    fig = fig_scatter_plot(df.loc[mask],freq,index_val,wnd)
    return(fig)

##########################################################################################################################################################################################
#                                                                                        page_4
##########################################################################################################################################################################################
@app.callback(
    Output('vol-dd-mvstd', 'options'),
    [Input('vol-dd-freq', 'value')]
)
def update_mvstd_dropdown_4(freq):
    if(freq == 'daily'):
        options=[{'label': str(i)+'D', 'value': int(i)} for i in [15,30,60,90]]
        print(options)
    elif(freq == '15min'):
        options=[{'label': str(i//60)+'hr', 'value': str(i)+'T'} for i in [60,120,240,720,1440]]
        print(options)
    return(options)
    
page_4_layout = html.Div([ get_layout_4() ])
    
@app.callback([Output('vol', 'figure'),
               Output('vol-focus-hk', 'figure'),
               Output('vol-focus-nikkei', 'figure'),
               Output('vol-focus-spmini500', 'figure'),
               Output('vol-focus-eurostoxx50', 'figure'),
               Output('vol-focus-vix', 'figure')],
              [Input('vol-dd-freq', 'value'),
               Input('vol-dd-index', 'value'),
               Input('vol-dd-mvstd', 'value')],
)              
def update_fig_4(freq,index_val,wnd):
    df = get_df_choice(freq,index_val,wnd)
    mask = (df.index >= pd.to_datetime('2020-03-01 00:00:00').tz_localize('UTC'))
    
    fig,fig_hk,fig_nikkei,fig_spmini500,fig_eustoxx50,fig_vix = fig_bar_plot(df.loc[mask],freq,index_val,wnd)
    return(fig,fig_hk,fig_nikkei,fig_spmini500,fig_eustoxx50,fig_vix)

##########################################################################################################################################################################################
#                                                                                        page_5
##########################################################################################################################################################################################
page_5_layout = html.Div([ get_layout_5() ])

@app.callback([Output('corr', 'figure'),
               Output('corr-table', 'children')],
              [Input('htpp-dd-freq','value'),
               Input('htpp-dd-thrs', 'value')]
)              
def update_fig_5(freq,vthreshold):
    if(freq == 'daily'):
        mask_hk = (df_hk_daily.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_hk_daily.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_hk_select = df_hk_daily.loc[mask_hk]

        mask_nikkei = (df_nikkei_daily.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_nikkei_daily.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_nikkei_select = df_nikkei_daily.loc[mask_nikkei]

        mask_spmini500 = (df_spmini500_daily.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_spmini500_daily.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_spmini500_select = df_spmini500_daily.loc[mask_spmini500]

        mask_eustoxx50 = (df_eustoxx50_daily.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_eustoxx50_daily.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_eustoxx50_select = df_eustoxx50_daily.loc[mask_eustoxx50]

        mask_vix= (df_vix_daily.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_vix_daily.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_vix_select = df_vix_daily.loc[mask_vix]
    elif(freq == '15min'):
        mask_hk = (df_hk_minute.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_hk_minute.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_hk_select = df_hk_minute.loc[mask_hk]

        mask_nikkei = (df_nikkei_minute.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_nikkei_minute.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_nikkei_select = df_nikkei_minute.loc[mask_nikkei]

        mask_spmini500 = (df_spmini500_minute.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_spmini500_minute.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_spmini500_select = df_spmini500_minute.loc[mask_spmini500]

        mask_eustoxx50 = (df_eustoxx50_minute.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_eustoxx50_minute.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_eustoxx50_select = df_eustoxx50_minute.loc[mask_eustoxx50]

        mask_vix= (df_vix_minute.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_vix_minute.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
        df_vix_select = df_vix_minute.loc[mask_vix]
        
    df_hk_select.columns = [ el+'_hk' for el in df_hk_select.columns ]
    df_nikkei_select.columns = [ el+'_nk' for el in df_nikkei_select.columns ]
    df_spmini500_select.columns = [ el+'_us' for el in df_spmini500_select.columns ]
    df_eustoxx50_select.columns = [ el+'_eu' for el in df_eustoxx50_select.columns ]
    df_vix_select.columns = [ el+'_vix' for el in df_vix_select.columns ]

    df_all = pd.concat([df_hk_select,df_nikkei_select,df_spmini500_select,df_eustoxx50_select,df_vix_select],axis=1).dropna()

    # correlation
    df_corr_all = df_all[[el for el in df_all.columns if 'H-L' not in el]]    
    fig_corr, corr_matrix_filtered = data_pairheat(df_corr_all,'Correlation matrix for the time series of all indices - Week 10th and 17th May',vthreshold)

    corr_matrix_dct = corr_matrix_filtered.unstack().to_dict()
    corr_matrix_dct = {k: v for k, v in corr_matrix_dct.items() if pd.Series(v).notna().all()}
    corr_matrix_dct_sorted = {k: v for k, v in sorted(corr_matrix_dct.items(), key=lambda x: x[1], reverse=True)}

    pairs_1 = [x[0] for x in list(corr_matrix_dct_sorted.keys())]
    pairs_2 = [x[1] for x in list(corr_matrix_dct_sorted.keys())]
    values = [ round(el,4) for el in list(corr_matrix_dct_sorted.values()) ]
    res = pd.DataFrame.from_dict({'pair_1': pairs_1,'pair_2': pairs_2,'correlation': values})  
    df_res_table = df_to_table(res)
    
    return(fig_corr,df_res_table)

##########################################################################################################################################################################################
#                                                                                        page_6
##########################################################################################################################################################################################
page_6_layout = html.Div([ get_layout_6() ])

@app.callback([Output('ret-dist-comp', 'figure'),
               Output('log-ret-dist-comp', 'figure')],
              [Input('ret-dd-idx-1', 'value'),
               Input('ret-dd-idx-2', 'value'),
               Input('ret-dd-day-1', 'value'),
               Input('ret-dd-month-1', 'value'),
               Input('ret-dd-day-2', 'value'),
               Input('ret-dd-month-2', 'value'),
               Input('ret-dd-ohlc-1', 'value'),
               Input('ret-dd-ohlc-2', 'value')]
)
def update_fig_6(index_val_1,index_val_2,day_1,month_1,day_2,month_2,ohlc_1,ohlc_2):
    date_1_low = pd.to_datetime('2020-'"{:02d}"'-'"{:02d}".format(month_1,day_1)+' 00:00:00') 
    date_1_up = pd.to_datetime('2020-'"{:02d}"'-'"{:02d}".format(month_1,day_1)+' 23:59:00') 
    date_2_low = pd.to_datetime('2020-'"{:02d}"'-'"{:02d}".format(month_2,day_2)+' 00:00:00')
    date_2_up = pd.to_datetime('2020-'"{:02d}"'-'"{:02d}".format(month_2,day_2)+' 23:59:00')
    
    mask_date_hk_1 = (df_hk_minute.index >= date_1_low.tz_localize('UTC')) & (df_hk_minute.index <= date_1_up.tz_localize('UTC'))
    mask_date_hk_2 = (df_hk_minute.index >= date_2_low.tz_localize('UTC')) & (df_hk_minute.index <= date_2_up.tz_localize('UTC'))
    
    mask_date_nk_1 = (df_nikkei_minute.index >= date_1_low.tz_localize('UTC')) & (df_nikkei_minute.index <= date_1_up.tz_localize('UTC'))
    mask_date_nk_2 = (df_nikkei_minute.index >= date_2_low.tz_localize('UTC')) & (df_nikkei_minute.index <= date_2_up.tz_localize('UTC'))

    mask_date_sp_1 = (df_spmini500_minute.index >= date_1_low.tz_localize('UTC')) & (df_spmini500_minute.index <= date_1_up.tz_localize('UTC'))
    mask_date_sp_2 = (df_spmini500_minute.index >= date_2_low.tz_localize('UTC')) & (df_spmini500_minute.index <= date_2_up.tz_localize('UTC'))

    mask_date_eu_1 = (df_eustoxx50_minute.index >= date_1_low.tz_localize('UTC')) & (df_eustoxx50_minute.index <= date_1_up.tz_localize('UTC'))
    mask_date_eu_2 = (df_eustoxx50_minute.index >= date_2_low.tz_localize('UTC')) & (df_eustoxx50_minute.index <= date_2_up.tz_localize('UTC'))

    mask_date_vix_1 = (df_vix_minute.index >= date_1_low.tz_localize('UTC')) & (df_vix_minute.index <= date_1_up.tz_localize('UTC'))
    mask_date_vix_2 = (df_vix_minute.index >= date_2_low.tz_localize('UTC')) & (df_vix_minute.index <= date_2_up.tz_localize('UTC'))
    
    df_hk_date_1 = df_hk_minute.loc[mask_date_hk_1]
    df_hk_date_2 = df_hk_minute.loc[mask_date_hk_2]

    df_nk_date_1 = df_nikkei_minute.loc[mask_date_nk_1]
    df_nk_date_2 = df_nikkei_minute.loc[mask_date_nk_2]

    df_sp_date_1 = df_spmini500_minute.loc[mask_date_sp_1]
    df_sp_date_2 = df_spmini500_minute.loc[mask_date_sp_2]

    df_eu_date_1 = df_eustoxx50_minute.loc[mask_date_eu_1]
    df_eu_date_2 = df_eustoxx50_minute.loc[mask_date_eu_2]

    df_vix_date_1 = df_vix_minute.loc[mask_date_vix_1]
    df_vix_date_2 = df_vix_minute.loc[mask_date_vix_2]
    
    fig1,fig2 = fig_dist_comp(index_val_1,index_val_2,df_hk_date_1,df_hk_date_2,
                            df_nk_date_1,df_nk_date_2,
                            df_sp_date_1,df_sp_date_2,
                            df_eu_date_1,df_eu_date_2,
                            df_vix_date_1,df_vix_date_2,
                            ohlc_1,ohlc_2)
    return(fig1,fig2)

##########################################################################################################################################################################################
#                                                                                        page_7
##########################################################################################################################################################################################
page_7_layout = html.Div([ get_layout_7() ])

@app.callback(Output('stats-table-1', 'children'),
              [Input('stats-ohlcv', 'value')]
)
def update_fig_7(ohlc):
    df_summary_1 = df_to_table(df_hk_minute[:10])
    table_stats_ohlcv(df_hk_minute,df_nikkei_minute,df_spmini500_minute,df_eustoxx50_minute,ohlc)
    return(df_summary_1)
    
####################################################################################################################################################################################
#                                                                                            page display                                                                          # 
####################################################################################################################################################################################
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return page_3_layout
    elif pathname == '/page-4':
        return page_4_layout
    elif pathname == '/page-5':
        return page_5_layout
    elif pathname == '/page-6':
        return page_6_layout
    elif pathname == '/page-7':
        return page_7_layout

if __name__ == '__main__':
    app.run_server(debug=True)

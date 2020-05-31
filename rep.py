#!/usr/bin/python3

import colorlover as cl
import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import datetime
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
from dt_read import DataProcessor
from pandas.plotting import register_matplotlib_converters
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import pdist
from statsmodels.sandbox.regression.predstd import wls_prediction_std

warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None 
register_matplotlib_converters()
   
color_1 = '#c0b8c3'
color_2 = '#3e75cf' # '#FF5733'
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

def get_data_all():
    obj_reader = DataProcessor('data_in','data_out','conf_model.yml')
    obj_reader.read_prm()
    obj_reader.process()
    return(obj_reader.hk_index, obj_reader.nikkei_index, obj_reader.spmini_index)

df_hk, df_nikkei, df_spmini500 = get_data_all()
    
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
    
def data_pairheat(df):
    df = df.astype(float)
    corr_matrix = df.corr()

    data = [dict(
        type = 'heatmap',
        z = corr_matrix.values,
        y = corr_matrix.columns,
        x = corr_matrix.index,
        name = 'Correlation matrix for the time series of all indices - Week 10th and 17th May',
        annotation_text = corr_matrix.round(2).values
        )]

    layout = dict()
    fig = dict(data=data,layout=layout)
    
    fig['layout'] = dict()
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['tickangle'] = -45
    fig['layout']['xaxis'] = dict(zeroline=False)
    fig['layout']['yaxis'] = dict(zeroline=False, autorange='reversed')
    fig['layout']['legend'] = dict(orientation = 'h', y=0.9, x=0.3, yanchor='right')
    fig['layout']['margin'] = dict(t=40, b=40, r=40, l=100)
    fig['layout']['width'] = 1200
    fig['layout']['height'] = 1200
    fig['layout']['autosize'] = True
    fig['layout']['text'] = corr_matrix.round(2).values
    fig['layout']['xgap'] = 3
    fig['layout']['ygap'] = 3
    fig['layout']['colorscale'] = [[0, 'rgba(255, 255, 255,0)'], [1, '#a3a7b0']]
    fig['layout']['thicknessmode'] = 'pixels'
    fig['layout']['thickness'] = 50
    fig['layout']['lenmode'] = 'pixels'
    fig['layout']['len'] =200    
    
    return(fig,corr_matrix)
   
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
    fig['layout']['colorscale'] = [[0, 'rgba(255, 255, 255,0)'], [1, '#a3a7b0']]
    fig['layout']['thicknessmode'] = 'pixels'
    fig['layout']['thickness'] = 50
    fig['layout']['lenmode'] = 'pixels'
    fig['layout']['len'] =200    

    return(fig)
   
def candlestick_plot(df,index_val):
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
        
    rangeselector=dict(
        visibe = True,
        x = 0, y = 0.9,
        bgcolor = 'rgba(150, 200, 250, 0.4)',
        font = dict( size = 16 ),
        buttons=list([
                      dict(count=1,
                           label='reset',
                           step='all'),
                      dict(count=1,
                           label='1mo',
                           step='month',
                           stepmode='backward'),
                      dict(count=5,
                           label='5day',
                           step='day',
                           stepmode='backward'),
                      dict(count=24,
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
                      dict(step='all')
                      ]))
    fig['layout']['xaxis']['rangeselector'] = rangeselector

    # fig['data'].append(dict(x=mv_x, y=mv_y, type='scatter', mode='lines', 
    #                         line = dict( width = 2 ),
    #                         marker = dict( color = color_3 ),
    #                         yaxis = 'y2', name='Moving Standard Deviation'+'_H-L'))

    colors = []

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
    
def nav_menu():
    nav = dbc.Nav(
        [
            dbc.NavLink("Raw plots", href='/page-1', id='page-1-link', style=STYLE_1),
            dbc.NavLink("Volume", href='/page-2', id='page-2-link', style=STYLE_1),
            dbc.NavLink("H-L & Volume", href='/page-3', id='page-3-link', style=STYLE_1),
            dbc.NavLink("H-L & Volatility", href='/page-4', id='page-4-link', style=STYLE_1),
            dbc.NavLink("Correlations", href='/page-5', id='page-5-link', style=STYLE_1),
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
                        html.Div(html.P([html.Br(),html.H5(html.B('For each chosen index, different metrics are shown. The moving average represented in the graph is based on the Close value. It is also possible to show for different backward windows. We notice that the most active sessions start from Mid-March for Nikkei225 and SP500 contracts and from beginning of May for Hang Seng')),html.Br(),html.H5(html.B('For Nikkei, the daily trend in terms of volume activity shows a peak of at the start of the trading session and gradual decrease towards the end of the day.')),html.Br(),html.H5(html.B('For Hang Seng, the daily trend in terms of volume activity shows a parabolic evolution from the start to the end of the day')),html.Br(),html.H5(html.B('For SP500, the daily trend in terms of volume activity shows a parabolic evolution from the start to the end of the day'))]), style=STYLE_8),
                        html.Div([
                                  html.Div(html.H4('Index choice'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ohlc-dd-index',
                                      options=[{'label': i, 'value': i} for i in ['Nikkei225','HangSeng','eMiniSP500']],
                                      value='Nikkei225',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
                        html.Div([
                                  html.Div(html.H4('Moving average time window'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ohlc-dd-mvavg',
                                      options=[{'label': str(i//60)+'hr', 'value': str(i)+'T'} for i in [60,120,240,720,1440]],
                                      value='60T',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
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
                        html.Div(html.P([html.Br(),html.H5(html.B('The methodology is to first cut out dates for which volume is insignificant by filtering on a specific date for each of the index. This is done by comparing the volume from the raw data plots')),html.Br(),html.H5(html.B('HangSeng: cut-off date = 2020-04-26')),html.Br(),html.H5(html.B('Nikkei225: cut-off date = 2020-03-08')),html.Br(),html.H5(html.B('eMiniSP500: cut-off date = 2020-03-08')),html.Br(),html.H5(html.B('After these cut-off dates the volumes have already spiked and got to tradable levels')),html.Br(),html.H5(html.B('Then we select the intersection of the previous cut-off dates, i.e. 2020-04-26'))]), style=STYLE_8)]),
              html.Div([
                        html.Div([
                                  html.Div(html.H4('Index choice'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ohlc-dd-index-cut',
                                      options=[{'label': i, 'value': i} for i in ['Nikkei225','HangSeng','eMiniSP500']],
                                      value='Nikkei225',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
                        html.Div([
                                  html.Div(html.H4('Moving average time window'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='ohlc-dd-mvavg-cut',
                                      options=[{'label': str(i//60)+'hr', 'value': str(i)+'T'} for i in [60,120,240,720,1440]],
                                      value='60T',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
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
                        html.Div(html.P([html.Br(),html.H5(html.B('Representation of High-Low prices focuses from the latest cut-off date of 2020-04-26 (for Hang Seng)')),html.Br(),html.H5(html.B('The Volume representation is relative to the movement of the High-Low price difference'))]), style=STYLE_8)]),
              html.Div([
                        html.Div([
                                  html.Div(html.H4('Index choice'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='hl-dd-index',
                                      options=[{'label': i, 'value': i} for i in ['Nikkei225','HangSeng','eMiniSP500']],
                                      value='Nikkei225',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
                        html.Div([
                                  html.Div(html.H4('Moving average time window'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='hl-dd-mvavg',
                                      options=[{'label': str(i//60)+'hr', 'value': str(i)+'T'} for i in [60,120,240,720,1440]],
                                      value='60T',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
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
                        html.Div(html.P([html.Br(),html.H5(html.B('As described in previous tabs, 2020-04-26 is the cut-off dates for which common trading sessions exist between the 3 indices considered here')),html.Br(),html.H5(html.B('As the week commencing 3rd May contains reduced trading days for Japan (due to holidays), we focus on two full weeks starting 10th May and 17th May')),html.Br(),html.H5(html.B('In addition these time windows are far enough from the volatile periods of the beginning of the year. This enables to look into a more general patterns/relationship between the different indices'))]), style=STYLE_8)]),
              html.Div([
                        html.Div([
                                  html.Div(html.H4('Index choice'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='vol-dd-index',
                                      options=[{'label': i, 'value': i} for i in ['Nikkei225','HangSeng','eMiniSP500']],
                                      value='Nikkei225',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
                        html.Div([
                                  html.Div(html.H4('Moving Standard Deviation time window'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='vol-dd-mvavg',
                                      options=[{'label': str(i//60)+'hr', 'value': str(i)+'T'} for i in [60,120,240,720,1440]],
                                      value='60T',
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
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
                                  dcc.Graph(
                                      id = 'corr',
                                      style=STYLE_4)
                                      ]),
                        html.Div([
                                  html.Div(html.H4('threshold correlation'),style=STYLE_6),
                                  dcc.Dropdown(
                                      id='htpp-dd-thrs',
                                      options=[{'label': i, 'value': i} for i in [round(el,2) for el in np.linspace(-1.0,1.0,21)]],
                                      value=0.5,
                                      style=STYLE_2
                                      )
                                      ],style=STYLE_3),
                        html.Img(id = 'data-pair',
                                 src = '',
                                 style=STYLE_4),
                        html.Div(
                            id='corr-table',
                            className='tableDiv'
                            ),
                        html.Div([
                                  html.Div(html.P([html.Br(),html.H2(html.B('Simple return for indices based on their correlation')),html.Br()]), style=STYLE_8),
                                  dcc.Graph(
                                      id = 'data-heatmap',
                                      style=STYLE_4)
                                      ]),
                        ]),
              ])
    return(html_res)

# ##########################################################################################################################################################################################
# #                                                                                        layout_6
# ##########################################################################################################################################################################################
# def get_layout():
#     html_res = \
#     html.Div([
#         html.Div([
#             html.Div(html.H6('Cluster Method'),style=STYLE_6),
#             dcc.Dropdown(
#                 id='method-dropdown',
#                 options=[{'label': i, 'value': i} for i in ['single','complete','average','weighted','centroid','median','ward']],
#                 value='ward',
#                 style=STYLE_2
#             )
#             ],style=STYLE_3),
#         html.Div([
#             html.Div(html.H6('Cluster Metric'),style=STYLE_6),
#             dcc.Dropdown(
#                 id='metric-dropdown',
#                 options=[{'label': i, 'value': i} for i in ['euclidean','correlation','cosine','dtw']],
#                 value='euclidean',
#                 style=STYLE_2
#             )
#             ],style=STYLE_3),
#         html.Div([
#             html.Div(html.H6('Cluster max #'),style=STYLE_6),
#             dcc.Dropdown(
#                 id='max-cluster-dropdown',
#                 options=[{'label': i, 'value': i} for i in range(int(params['max_cluster_rep']))],
#                 value='17',
#                 style=STYLE_2
#             )
#             ],style=STYLE_3),
#         html.Div([
#             html.Div(html.P([html.Br(),html.H2(html.B('Cluster dendrogram - Clustered time series')),html.Br()]), style=STYLE_5),
#             html.Img(id = 'cluster-plot',
#                            src = '',
#                            style=STYLE_4)
#         ]),
#         html.Div([
#             html.Div(html.P([html.Br(),html.H2(html.B('Clustered time series')),html.Br()]), style=STYLE_5),
#             html.Img(id = 'qty-plot',
#                            src = '',
#                            style=STYLE_4)
#         ]),
#         html.Div([
#             html.Div(html.P([html.Br(),html.H2(html.B('Cumulative return')),html.Br()]), style=STYLE_5),
#             html.Div(html.H6('Cluster Selected'),style=STYLE_6),
#             dcc.Dropdown(
#                 id='selected-cluster-dropdown',
#                 value='5',
#                 style=STYLE_2
#             )
#         ]),
#         html.Div([
#             dcc.Graph(
#                 id = 'qty-uniq-plot',
#                 style=STYLE_4)
#         ]),
#         html.Div(
#             id='cluster-table',
#             className='tableDiv'
#         ),
#         html.Div([
#             html.Div(html.P([html.Br(),html.H2(html.B('Correlation matrix from selected cluster')),html.Br()]), style=STYLE_5),
#         ]),
#         html.Div([
#             dcc.Graph(
#                 id = 'corr-uniq-plot',
#                 style=STYLE_7)
#         ]),
#         html.Div([
#             html.Div(html.P([html.Br(),html.H2(html.B('Pairs plot from selected cluster')),html.Br()]), style=STYLE_5),
#             html.Img(id = 'pairplot-uniq-plot',
#                            src = '',
#                            style=STYLE_4)
#         ]),
#         html.Div([
#             html.Div(html.P([html.Br(),html.H2(html.B('Dynamic time warping from selected cluster')),html.Br()]), style=STYLE_5),
#             html.Img(id = 'dtws-uniq-plot',
#                            src = '',
#                            style=STYLE_4)
#         ]),
#         html.Div(html.P([html.Br(),html.H2(html.B('Cluster counts')),html.Br()]), style=STYLE_5),
#         html.Div([html.Img(id = 'clusters-hist-plot',
#                            src = '',
#                            style=STYLE_4)
#         ]),
#         html.Div(html.P([html.Br(),html.H2(html.B('Components mapping from dendrogram')),html.Br()]), style=STYLE_5),
#         html.Div([html.Img(id = 'clusters-map-text-plot',
#                            src = '',
#                            style=STYLE_4)
#         ]),
#         html.Div(html.P([html.Br(),html.H2(html.B('Mapping from selected cluster')),html.Br()]), style=STYLE_5),
#         html.Div([html.Img(id = 'clusters-uniq-plot',
#                            src = '',
#                            style=STYLE_4)
#         ]),
#         html.Div(html.P([html.Br(),html.H2(html.B('Distributions of time series from selected cluster')),html.Br()]), style=STYLE_5),
#         html.Div([html.Img(id = 'klb-uniq-plot',
#                            src = '',
#                            style=STYLE_4)
#         ]),
#     ])
#     return(html_res)
    
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
    [Output(f"page-{i}-link", "active") for i in range(1,6)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        return True, False, False, False, False
    return [pathname == f"/page-{i}" for i in range(1,6)]

##########################################################################################################################################################################################
#                                                                                        page_1
##########################################################################################################################################################################################
page_1_layout = html.Div([ get_layout_1() ])

@app.callback(Output('ohlc', 'figure'),
              [Input('ohlc-dd-index', 'value'),
               Input('ohlc-dd-mvavg', 'value'),],
)
def update_fig_1(index_val,wnd):
    if(index_val=='HangSeng'):
        df = df_hk
    elif(index_val=='Nikkei225'):
        df = df_nikkei
    elif(index_val=='eMiniSP500'):
        df = df_spmini500

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
        
    rangeselector=dict(
        visibe = True,
        x = 0, y = 0.9,
        bgcolor = 'rgba(150, 200, 250, 0.4)',
        font = dict( size = 16 ),
        buttons=list([
                      dict(count=1,
                           label='reset',
                           step='all'),
                      dict(count=1,
                           label='1mo',
                           step='month',
                           stepmode='backward'),
                      dict(count=5,
                           label='5day',
                           step='day',
                           stepmode='backward'),
                      dict(count=24,
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
                      dict(step='all')
                      ]))
    fig['layout']['xaxis']['rangeselector'] = rangeselector

    mv_y = movingaverage(df['Close'],window_size=wnd)
    mv_x = list(df.index)

    # Clip the ends
    mv_x = mv_x[5:-5]
    mv_y = mv_y[5:-5]

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
    
##########################################################################################################################################################################################
#                                                                                        page_2
##########################################################################################################################################################################################
page_2_layout = html.Div([ get_layout_2() ])

@app.callback(Output('ohlc-cut', 'figure'),
              [Input('ohlc-dd-index-cut', 'value'),
               Input('ohlc-dd-mvavg-cut', 'value'),],
)
def update_fig_2(index_val,wnd):
    if(index_val=='HangSeng'):
        df = df_hk
        mask = (df.index >= pd.to_datetime('2020-04-26 00:00:00').tz_localize('UTC'))
    elif(index_val=='Nikkei225'):
        df = df_nikkei
        mask = (df.index >= pd.to_datetime('2020-04-26 00:00:00').tz_localize('UTC'))
    elif(index_val=='eMiniSP500'):
        df = df_spmini500
        mask = (df.index >= pd.to_datetime('2020-04-26 00:00:00').tz_localize('UTC'))

    df = df.loc[mask]

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
        
    rangeselector=dict(
        visibe = True,
        x = 0, y = 0.9,
        bgcolor = 'rgba(150, 200, 250, 0.4)',
        font = dict( size = 16 ),
        buttons=list([
                      dict(count=1,
                           label='reset',
                           step='all'),
                      dict(count=1,
                           label='1mo',
                           step='month',
                           stepmode='backward'),
                      dict(count=5,
                           label='5day',
                           step='day',
                           stepmode='backward'),
                      dict(count=24,
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
                      dict(step='all')
                      ]))
    fig['layout']['xaxis']['rangeselector'] = rangeselector

    mv_y = movingaverage(df['Close'],window_size=wnd)
    mv_x = list(df.index)

    # Clip the ends
    mv_x = mv_x[5:-5]
    mv_y = mv_y[5:-5]

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

##########################################################################################################################################################################################
#                                                                                        page_3
##########################################################################################################################################################################################
page_3_layout = html.Div([ get_layout_3() ])

@app.callback(Output('hl', 'figure'),
              [Input('hl-dd-index', 'value'),
               Input('hl-dd-mvavg', 'value'),],
)              
def update_fig_3(index_val,wnd):
    if(index_val=='HangSeng'):
        df = df_hk
        mask = (df.index >= pd.to_datetime('2020-04-26 00:00:00').tz_localize('UTC'))
    elif(index_val=='Nikkei225'):
        df = df_nikkei
        mask = (df.index >= pd.to_datetime('2020-04-26 00:00:00').tz_localize('UTC'))
    elif(index_val=='eMiniSP500'):
        df = df_spmini500
        mask = (df.index >= pd.to_datetime('2020-04-26 00:00:00').tz_localize('UTC'))

    df = df.loc[mask]

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
        
    rangeselector=dict(
        visibe = True,
        x = 0, y = 0.9,
        bgcolor = 'rgba(150, 200, 250, 0.4)',
        font = dict( size = 16 ),
        buttons=list([
                      dict(count=1,
                           label='reset',
                           step='all'),
                      dict(count=1,
                           label='1mo',
                           step='month',
                           stepmode='backward'),
                      dict(count=5,
                           label='5day',
                           step='day',
                           stepmode='backward'),
                      dict(count=24,
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
                      dict(step='all')
                      ]))
    fig['layout']['xaxis']['rangeselector'] = rangeselector

    mv_y = movingaverage(df['H-L'],window_size=wnd)
    mv_x = list(df.index)

    # Clip the ends
    mv_x = mv_x[5:-5]
    mv_y = mv_y[5:-5]

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

##########################################################################################################################################################################################
#                                                                                        page_4
##########################################################################################################################################################################################
page_4_layout = html.Div([ get_layout_4() ])

@app.callback([Output('vol', 'figure'),
               Output('vol-focus-hk', 'figure'),
               Output('vol-focus-nikkei', 'figure'),
               Output('vol-focus-spmini500', 'figure')],
              [Input('vol-dd-index', 'value'),
               Input('vol-dd-mvavg', 'value')],
)              
def update_fig_4(index_val,wnd):
    if(index_val=='HangSeng'):
        df = df_hk
        mask = (df.index >= pd.to_datetime('2020-04-26 00:00:00').tz_localize('UTC'))
    elif(index_val=='Nikkei225'):
        df = df_nikkei
        mask = (df.index >= pd.to_datetime('2020-04-26 00:00:00').tz_localize('UTC'))
    elif(index_val=='eMiniSP500'):
        df = df_spmini500
        mask = (df.index >= pd.to_datetime('2020-04-26 00:00:00').tz_localize('UTC'))

    df = df.loc[mask]

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
        
    rangeselector=dict(
        visibe = True,
        x = 0, y = 0.9,
        bgcolor = 'rgba(150, 200, 250, 0.4)',
        font = dict( size = 16 ),
        buttons=list([
                      dict(count=1,
                           label='reset',
                           step='all'),
                      dict(count=1,
                           label='1mo',
                           step='month',
                           stepmode='backward'),
                      dict(count=5,
                           label='5day',
                           step='day',
                           stepmode='backward'),
                      dict(count=24,
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
                      dict(step='all')
                      ]))
    fig['layout']['xaxis']['rangeselector'] = rangeselector

    mv_y = movingstd(df['H-L'],window_size=wnd)
    mv_x = list(df.index)

    # Clip the ends
    mv_x = mv_x[5:-5]
    mv_y = mv_y[5:-5]

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

    mask_hk = (df_hk.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_hk.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
    df_hk_select = df_hk.loc[mask_hk]
    fig_hk = candlestick_plot(df_hk_select,'HangSeng')

    mask_nikkei = (df_nikkei.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_nikkei.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
    df_nikkei_select = df_nikkei.loc[mask_nikkei]
    fig_nikkei = candlestick_plot(df_nikkei_select,'Nikkei225')

    mask_spmini500 = (df_spmini500.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_spmini500.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
    df_spmini500_select = df_spmini500.loc[mask_spmini500]
    fig_spmini500 = candlestick_plot(df_spmini500_select,'eMiniSP500')
    
    return(fig,fig_hk,fig_nikkei,fig_spmini500)

##########################################################################################################################################################################################
#                                                                                        page_5
##########################################################################################################################################################################################
page_5_layout = html.Div([ get_layout_5() ])

@app.callback([Output('corr', 'figure'),
               Output('corr-table', 'children'),
               Output('data-heatmap', 'figure'),
               Output('data-pair', 'src')],
              [Input('htpp-dd-thrs', 'value')]
)              
def update_fig_5(vthreshold):
    mask_hk = (df_hk.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_hk.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
    df_hk_select = df_hk.loc[mask_hk]
    mask_nikkei = (df_nikkei.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_nikkei.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
    df_nikkei_select = df_nikkei.loc[mask_nikkei]
    mask_spmini500 = (df_spmini500.index >= pd.to_datetime('2020-05-09 00:00:00').tz_localize('UTC')) & (df_spmini500.index < pd.to_datetime('2020-05-23 00:00:00').tz_localize('UTC'))
    df_spmini500_select = df_spmini500.loc[mask_spmini500]

    df_hk_select.columns = [ el+'_hk' for el in df_hk_select.columns ]
    df_nikkei_select.columns = [ el+'_nk' for el in df_nikkei_select.columns ]
    df_spmini500_select.columns = [ el+'_us' for el in df_spmini500_select.columns ]

    df_all = pd.concat([df_hk_select,df_nikkei_select,df_spmini500_select],axis=1).dropna()

    # correlation
    df_corr_all = df_all[[el for el in df_all.columns if 'H-L' not in el]]    
    fig_corr, corr_matrix = data_pairheat(df_corr_all)
    fig_corr_pair = data_pairplot(df_corr_all)

    corr_matrix_dct = corr_matrix.unstack().to_dict()
    corr_matrix_dct_filter = {k: v for k, v in sorted(corr_matrix_dct.items(), key=lambda x: x[1], reverse=True)}
    corr_matrix_dct_filter = {k: v for k, v in corr_matrix_dct_filter.items() if v < 1.0 and v >= vthreshold}
    pairs_1 = [x[0] for x in list(corr_matrix_dct_filter.keys())]
    pairs_2 = [x[1] for x in list(corr_matrix_dct_filter.keys())]
    values = [ round(el,4) for el in list(corr_matrix_dct_filter.values()) ]
    df_res_table = df_to_table(pd.DataFrame.from_dict({'pair_1': pairs_1,'pair_2': pairs_2,'correlation': values}))

    selected_headers = list(set(list(sum(corr_matrix_dct_filter.keys(),()))))
    fig_heatmap = data_heatmap(df_all[[el for el in selected_headers if 'H-L' not in el]].pct_change())
    
    return(fig_corr,df_res_table,fig_heatmap,fig_corr_pair)
        
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

if __name__ == '__main__':
    app.run_server(debug=True)

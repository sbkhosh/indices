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
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy
import scipy.cluster.hierarchy as hac
import statsmodels.api as sm
import sys, os
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
   
color_1 = '#87188D'
STYLE_1 = {'font-family': 'Calibri','font-size':50,'width': '33%','display':'inline-block'}
STYLE_2 = {'font-family': 'Calibri','font-size':15,'width': '50%','display':'inline-block'}
STYLE_3 = {'width': '25%', 'float': 'left', 'display': 'inline-block'}
STYLE_4 = {'height': '100%', 'width': '100%', 'float': 'left', 'padding': 90}
STYLE_5 = {'font-family': 'Calibri', 'color': color_1}
STYLE_6 = {'color': color_1}
STYLE_7 = {'height': '50%', 'width': '100%'}

def get_data_all():
    obj_reader = DataProcessor('data_in','data_out','conf_model.yml')
    obj_reader.read_prm()
    obj_reader.process()
    return(obj_reader.hk_index, obj_reader.nikkei_index, obj_reader.spmini_index)

df_hk, df_nikkei, df_spmini500 = get_data_all()

def nav_menu():
    nav = dbc.Nav(
        [
            dbc.NavLink("Raw plots", href='/page-1', id='page-1-link', style=STYLE_1),
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

def get_layout():
    html_res = \
    html.Div([
        html.Div([
            html.Div(html.H2('OHLC'),style=STYLE_6),
            dcc.Dropdown(
                id='ohlc-dropdown',
                options=[{'label': i, 'value': i} for i in ['Nikkei225','HangSeng','eMiniSP500']],
                value='Nikkei225',
                style=STYLE_2
            )
            ],style=STYLE_3),
        html.Div([
            dcc.Graph(
                id = 'ohlc',
                style=STYLE_4)
        ]),
        html.Div([
            html.Div(html.H2('Volume'),style=STYLE_6),
            dcc.Dropdown(
                id='volume-dropdown',
                options=[{'label': i, 'value': i} for i in ['Nikkei225','HangSeng','eMiniSP500']],
                value='Nikkei225',
                style=STYLE_2
            )
            ],style=STYLE_3),
        html.Div([
            dcc.Graph(
                id = 'volume',
                style=STYLE_4)
        ]),
        html.Div([
            html.Div(html.H2('OHLC - All'),style=STYLE_6),
            ],style=STYLE_3),
        html.Div([
            dcc.Graph(
                id = 'ohlc-all',
                style=STYLE_4)
        ]),
        html.Div([
            html.Div(html.H2('Volume - All'),style=STYLE_6),
            ],style=STYLE_3),
        html.Div([
            dcc.Graph(
                id = 'volume-all',
                style=STYLE_4)
        ]),
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
    [Output(f"page-{i}-link", "active") for i in range(1)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1)]

##########################################################################################################################################################################################
#                                                                                        page_1
##########################################################################################################################################################################################
page_1_layout = html.Div([ get_layout() ])

@app.callback(Output('ohlc', 'figure'),
              [Input('ohlc-dropdown', 'value')],
)
def update_fig_1(index_val):
    # df_hk, df_nikkei, df_spmini500 = get_data_all()
    
    if(index_val=='HangSeng'):
        data = df_hk
    elif(index_val=='Nikkei225'):
        data = df_nikkei
    elif(index_val=='eMiniSP500'):
        data = df_spmini500

    trace = go.Ohlc(x=data.index,
                    open=data['Open']/data['Open'].iloc[0],
                    high=data['High']/data['High'].iloc[0],
                    low=data['Low']/data['Low'].iloc[0],
                    close=data['Close']/data['Close'].iloc[0])
    
    return({
        'data': [trace],
        'layout': {
            'title': index_val,
            'xaxis': dict(rangeslider=dict(visible=False),tickformat='%Y-%m-%d %H:%M:%S'),
            'yaxis': dict(autoscale=True),
            }
        })

@app.callback(Output('volume', 'figure'),
              [Input('volume-dropdown', 'value')],
)
def update_fig_2(index_val):
    # df_hk, df_nikkei, df_spmini500 = get_data_all()
    
    if(index_val=='HangSeng'):
        data = df_hk
    elif(index_val=='Nikkei225'):
        data = df_nikkei
    elif(index_val=='eMiniSP500'):
        data = df_spmini500

    trace = go.Bar(x=data.index, y=data['Volume'])
    return({
        'data': [trace],
        'layout': {
            'title': index_val,
            'xaxis': dict(rangeslider=dict(visible=False),tickformat='%Y-%m-%d %H:%M:%S'),
            'yaxis': dict(autoscale=True),
            }
        })
       
@app.callback(Output('ohlc-all', 'figure'),
              [Input('ohlc-dropdown', 'value')],
)
def update_fig_3(index_val):
    # df_hk, df_nikkei, df_spmini500 = get_data_all()

    hovertext_1,hovertext_2,hovertext_3 = [],[],[]
    
    for i in range(len(df_hk['Open'])):
        hovertext_1.append('Date:'+str(df_hk.index[i])+
                           '<br>Open HangSeng: '+str(df_hk['Open'][i]/df_hk['Open'].iloc[0])+
                           '<br>High HangSeng: '+str(df_hk['High'][i]/df_hk['High'].iloc[0])+
                           '<br>Low HangSeng: '+str(df_hk['Low'][i]/df_hk['Low'].iloc[0])+
                           '<br>Close HangSeng: '+str(df_hk['Close'][i]/df_hk['Close'].iloc[0]))

    for i in range(len(df_nikkei['Open'])):
        hovertext_2.append('Date:'+str(df_nikkei.index[i])+
                           '<br>Open Nikkei225: '+str(df_nikkei['Open'][i]/df_nikkei['Open'].iloc[0])+
                           '<br>High Nikkei225: '+str(df_nikkei['High'][i]/df_nikkei['High'].iloc[0])+
                           '<br>Low Nikkei225: '+str(df_nikkei['Low'][i]/df_nikkei['Low'].iloc[0])+
                           '<br>Close Nikkei225: '+str(df_nikkei['Close'][i]/df_nikkei['Close'].iloc[0]))

    for i in range(len(df_spmini500['Open'])):
        hovertext_3.append('Date:'+str(df_spmini500.index[i])+
                           '<br>Open eMiniSP500: '+str(df_spmini500['Open'][i]/df_spmini500['Open'].iloc[0])+
                           '<br>High eMiniSP500: '+str(df_spmini500['High'][i]/df_spmini500['High'].iloc[0])+
                           '<br>Low eMiniSP500: '+str(df_spmini500['Low'][i]/df_spmini500['Low'].iloc[0])+
                           '<br>Close eMiniSP500: '+str(df_spmini500['Close'][i]/df_spmini500['Close'].iloc[0]))
        
    trace_1 = go.Ohlc(x=df_hk.index,
                      open=df_hk['Open']/df_hk['Open'].iloc[0],
                      high=df_hk['High']/df_hk['High'].iloc[0],
                      low=df_hk['Low']/df_hk['Low'].iloc[0],
                      close=df_hk['Close']/df_hk['Close'].iloc[0],
                      increasing_line_color= 'cyan', decreasing_line_color= 'gray',
                      text=hovertext_1,
                      hoverinfo='text',
                      name='HangSeng')

    trace_2 = go.Ohlc(x=df_nikkei.index,
                      open=df_nikkei['Open']/df_nikkei['Open'].iloc[0],
                      high=df_nikkei['High']/df_nikkei['High'].iloc[0],
                      low=df_nikkei['Low']/df_nikkei['Low'].iloc[0],
                      close=df_nikkei['Close']/df_nikkei['Close'].iloc[0],
                      increasing_line_color= 'green', decreasing_line_color= 'red',
                      text=hovertext_2,
                      hoverinfo='text',
                      name='Nikkei225')

    trace_3 = go.Ohlc(x=df_spmini500.index,
                      open=df_spmini500['Open']/df_spmini500['Open'].iloc[0],
                      high=df_spmini500['High']/df_spmini500['High'].iloc[0],
                      low=df_spmini500['Low']/df_spmini500['Low'].iloc[0],
                      close=df_spmini500['Close']/df_spmini500['Close'].iloc[0],
                      increasing_line_color= 'yellow', decreasing_line_color= 'black',
                      text=hovertext_3,
                      hoverinfo='text',
                      name='eMiniSP500')
    
    return({
        'data': [trace_1,trace_2,trace_3],
        'layout': {
            'title': '',
            'xaxis': dict(rangeslider=dict(visible=False),tickformat='%Y-%m-%d %H:%M:%S'),
            'yaxis': dict(autoscale=True),
            }
        })

@app.callback(Output('volume-all', 'figure'),
              [Input('volume-dropdown', 'value')],
)
def update_fig_4(index_val):
    # df_hk, df_nikkei, df_spmini500 = get_data_all()

    hovertext_1,hovertext_2,hovertext_3 = [],[],[]
    
    for i in range(len(df_hk['Volume'])):
        hovertext_1.append('Date:'+str(df_hk.index[i])+
                           '<br>Volume eMiniSP500: '+str(df_hk['Volume'][i]))
        
    for i in range(len(df_nikkei['Open'])):
        hovertext_2.append('Date:'+str(df_nikkei.index[i])+
                           '<br>Volume Nikkei225: '+str(df_nikkei['Volume'][i]))
        
    for i in range(len(df_spmini500['Open'])):
        hovertext_3.append('Date:'+str(df_spmini500.index[i])+
                           '<br>Volume eMiniSP500: '+str(df_spmini500['Volume'][i]))
        
    trace_1 = go.Bar(x=df_hk.index,
                     y=df_hk['Volume'],
                     text=hovertext_1,
                     hoverinfo='text',
                     name='HangSeng')
    trace_2 = go.Bar(x=df_nikkei.index,
                     y=df_nikkei['Volume'],
                     text=hovertext_2,
                     hoverinfo='text',
                     name='Nikkei225')
    trace_3 = go.Bar(x=df_spmini500.index,
                     y=df_spmini500['Volume'],
                     text=hovertext_3,
                     hoverinfo='text',
                     name='eMiniSP500')
                     
    return({
        'data': [trace_1,trace_2,trace_3],
        'layout': {
            'title': '',
            'xaxis': dict(rangeslider=dict(visible=False),tickformat='%Y-%m-%d %H:%M:%S'),
            'yaxis': dict(autoscale=False),
            }
        })
    
####################################################################################################################################################################################
#                                                                                            page display                                                                          # 
####################################################################################################################################################################################
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout

if __name__ == '__main__':
    app.run_server(debug=True)

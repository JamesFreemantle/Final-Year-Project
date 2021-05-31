#!/usr/bin/env python
# coding: utf-8

# In[1]:


#reading in csv downloaded from https://coinmetrics.io/community-network-data/
import pandas as pd
df = pd.read_csv('C:/College/Final Year Project/btccoinmetrics.csv')


# In[2]:


print(list(df))


# In[3]:


#these are the indicators I am intrested in, removing other columns from dataframe
df = df[['date','PriceUSD','NVTAdj90','CapMVRVCur','FeeTotNtv']]
df.index = df['date']
#starting the dataframe from the start of 2012 so that a fair comparison can be made to other indicators
df = df[1093:]
df.index=pd.DatetimeIndex(df.index)


# In[4]:


df


# In[5]:


#adding a column to the data to specify which halving cycle the data is in

halving_2012 = pd.to_datetime('2012-11-28')
halving_2016 = pd.to_datetime('2016-07-09')
halving_2020 = pd.to_datetime('2020-05-11')

cycle1 = df.loc[df.index < halving_2012]
cycle2 = df.loc[(df.index < halving_2016) & (df.index > halving_2012)]
cycle3 = df.loc[(df.index < halving_2020) & (df.index > halving_2016)]
cycle4 = df.loc[df.index > halving_2020]

cycle1['Cycle'] = '2012'
cycle2['Cycle'] = '2016'
cycle3['Cycle'] = '2020'
cycle4['Cycle'] = '2024'

frames = [cycle1,cycle2,cycle3,cycle4]
df = pd.concat(frames,sort = False)


# # MVRV Ratio

# In[6]:


#plotting the MVRV ratio with the Price
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np

fig1 =go.Figure([go.Scatter(x=df.index, y = df['CapMVRVCur'],name = 'MVRV')])
fig2 =go.Figure([go.Scatter(x=df.index, y = df['PriceUSD'],name = 'BTC Price')])

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(fig2.data[0], secondary_y=False)
fig.add_trace(fig1.data[0], secondary_y=True)

                 
fig.add_shape(type="line",
    x0='2020-05-11', y0=1, x1='2020-05-11', y1=100000,
    line=dict(color="MediumPurple",width=3,dash = 'dot'))
              
fig.add_shape(type="line",
    x0='2016-07-09', y0=1, x1='2016-07-09', y1=100000,
    line=dict(color="MediumPurple",width=3,dash = 'dot'))

fig.add_shape(type="line",
    x0='2012-11-28', y0=1, x1='2012-11-28', y1=100000,
    line=dict(color="MediumPurple",width=3,dash = 'dot'))

fig.add_trace(go.Scatter(
    x=['2020-05-11', '2016-07-09', '2012-11-28'],
    y=[70000,40000,20000],
    text=['Third Halving',
         'Second Halving',
          'First Halving'],
    mode="text",
))


fig.update_layout(xaxis = dict(title = 'Year'),
                  yaxis=dict(title = 'BTC Price',type="log", range=[np.log10(1), np.log10(100000)]), 
                  yaxis2=dict(title = 'MVRV Capitalisation'))

py.iplot(fig)


# In[7]:


#chceking the correlaion between the MVRV ratios across halving cycles
correlation_frame =pd.DataFrame()
correlation_frame['2020 Cycle'] = pd.Series(df['CapMVRVCur'].loc[df['Cycle']=='2020'].values)
correlation_frame['2012 Cycle'] = pd.Series(df['CapMVRVCur'].loc[df['Cycle']=='2012'].values)
correlation_frame['2016 Cycle'] = pd.Series(df['CapMVRVCur'].loc[df['Cycle']=='2016'].values)
correlation_frame['2024 Cycle'] = pd.Series(df['CapMVRVCur'].loc[df['Cycle']=='2024'].values)

correlation_frame.corr(method='pearson')
#correlation_frame


# In[8]:


#the MVRV ratio correlation to price
correlation_frame = pd.DataFrame()
correlation_frame['MVRV'] = df['CapMVRVCur']
correlation_frame['Price'] = df['PriceUSD']

correlation_frame.corr(method='pearson')


# # NVT Signal 

# In[9]:


# Plotting NVT Signal to price
fig1 =go.Figure([go.Scatter(x=df.index, y = df['NVTAdj90'],name = 'NVT')])
fig2 =go.Figure([go.Scatter(x=df.index, y = df['PriceUSD'],name = 'BTC Price')])

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(fig2.data[0], secondary_y=False)
fig.add_trace(fig1.data[0], secondary_y=True)

                 
fig.add_shape(type="line",
    x0='2020-05-11', y0=1, x1='2020-05-11', y1=100000,
    line=dict(color="MediumPurple",width=3,dash = 'dot'))
              
fig.add_shape(type="line",
    x0='2016-07-09', y0=1, x1='2016-07-09', y1=100000,
    line=dict(color="MediumPurple",width=3,dash = 'dot'))

fig.add_shape(type="line",
    x0='2012-11-28', y0=1, x1='2012-11-28', y1=100000,
    line=dict(color="MediumPurple",width=3,dash = 'dot'))

fig.add_trace(go.Scatter(
    x=['2020-05-11', '2016-07-09', '2012-11-28'],
    y=[70000,40000,20000],
    text=['Third Halving',
         'Second Halving',
          'First Halving'],
    mode="text",
))


fig.update_layout(xaxis = dict(title = 'Year'),
                  yaxis=dict(title = 'BTC Price',type="log", range=[np.log10(1), np.log10(100000)]), 
                  yaxis2=dict(title = 'NVT'))

py.iplot(fig)


# In[10]:


# checking correlation between NVT signals across halving cycles
correlation_frame =pd.DataFrame()
correlation_frame['2020 Cycle'] = pd.Series(df['NVTAdj90'].loc[df['Cycle']=='2020'].values)
correlation_frame['2012 Cycle'] = pd.Series(df['NVTAdj90'].loc[df['Cycle']=='2012'].values)
correlation_frame['2016 Cycle'] = pd.Series(df['NVTAdj90'].loc[df['Cycle']=='2016'].values)
correlation_frame['2024 Cycle'] = pd.Series(df['NVTAdj90'].loc[df['Cycle']=='2024'].values)

correlation_frame.corr(method='pearson')
#correlation_frame


# In[11]:


# correlation between NVT signal to MVRV ratio to Price
correlation_frame = pd.DataFrame()
correlation_frame['MVRV'] = df['CapMVRVCur']
correlation_frame['NVT'] = df['NVTAdj90']
correlation_frame['Price'] = df['PriceUSD']

correlation_frame.corr(method='pearson')


# # Fee per Day

# In[12]:


#checking correlation of the total Fees from blocks in a day to the price

fig1 =go.Figure([go.Scatter(x=df.index, y = df['FeeTotNtv'],name = 'Fee Per Day')])
fig2 =go.Figure([go.Scatter(x=df.index, y = df['PriceUSD'],name = 'BTC Price')])

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(fig2.data[0], secondary_y=False)
fig.add_trace(fig1.data[0], secondary_y=True)

                 
fig.add_shape(type="line",
    x0='2020-05-11', y0=1, x1='2020-05-11', y1=100000,
    line=dict(color="MediumPurple",width=3,dash = 'dot'))
              
fig.add_shape(type="line",
    x0='2016-07-09', y0=1, x1='2016-07-09', y1=100000,
    line=dict(color="MediumPurple",width=3,dash = 'dot'))

fig.add_shape(type="line",
    x0='2012-11-28', y0=1, x1='2012-11-28', y1=100000,
    line=dict(color="MediumPurple",width=3,dash = 'dot'))

fig.add_trace(go.Scatter(
    x=['2020-05-11', '2016-07-09', '2012-11-28'],
    y=[70000,40000,20000],
    text=['Third Halving',
         'Second Halving',
          'First Halving'],
    mode="text",
))


fig.update_layout(xaxis = dict(title = 'Year'),
                  yaxis=dict(title = 'BTC Price',type="log", range=[np.log10(1), np.log10(100000)]), 
                  yaxis2=dict(title = 'Fee per Day',type="log", range=[np.log10(1), np.log10(100000)]))

py.iplot(fig)


# In[13]:


#checking correlation between NVT ratio across halving cycles.

correlation_frame =pd.DataFrame()
correlation_frame['2020 Cycle'] = pd.Series(df['FeeTotNtv'].loc[df['Cycle']=='2020'].values)
correlation_frame['2012 Cycle'] = pd.Series(df['FeeTotNtv'].loc[df['Cycle']=='2012'].values)
correlation_frame['2016 Cycle'] = pd.Series(df['FeeTotNtv'].loc[df['Cycle']=='2016'].values)
correlation_frame['2024 Cycle'] = pd.Series(df['FeeTotNtv'].loc[df['Cycle']=='2024'].values)

correlation_frame.corr(method='pearson')
#correlation_frame


# In[14]:


#correlation between MVRV ratio / NVT signal / Block fees per day / Price
correlation_frame = pd.DataFrame()
correlation_frame['MVRV'] = df['CapMVRVCur']
correlation_frame['NVT'] = df['NVTAdj90']
correlation_frame['Fee'] = df['FeeTotNtv']
correlation_frame['Price'] = df['PriceUSD']

correlation_frame.corr(method='pearson')


# In[15]:


#Writing this new df to a csv so it can be used for machine learning in a different notebook
df.to_csv('C:/College/Final Year Project/new_indicators.csv')


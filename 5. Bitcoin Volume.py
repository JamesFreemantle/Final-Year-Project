#!/usr/bin/env python
# coding: utf-8

# In[1]:


# reading in csv files
import pandas as pd
df_daily = pd.read_csv('C:/College/Final Year Project/daily_cleaned.csv')
df_daily = df_daily.set_index('Date')
df_daily.index=pd.DatetimeIndex(df_daily.index)

df_weekly = pd.read_csv('C:/College/Final Year Project/weekly_cleaned.csv')
df_weekly = df_weekly.set_index('Unnamed: 0')
df_weekly.index.rename('Date',inplace = True)
df_weekly.index=pd.DatetimeIndex(df_weekly.index)

df_monthly = pd.read_csv('C:/College/Final Year Project/monthly_cleaned.csv')
df_monthly = df_monthly.set_index('Unnamed: 0')
df_monthly.index.rename('Date',inplace = True)
df_monthly.index=pd.DatetimeIndex(df_monthly.index)

# removing first few rows of the dataframes, I found that since indicator values are built on previous data,
# and there is no previous data to produce the firt indicator values, they defaulted to 100. These indicator values at 100 were removed. 

df_daily = df_daily.iloc[4:]
df_weekly = df_weekly.iloc[4:]
df_monthly = df_monthly.iloc[2:]


# In[2]:


import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

#plotting the price against the volatility volume ratio
fig1 =go.Figure([go.Scatter(x=df_monthly.index, y = df_monthly['weighted_price'],name = 'BTC Price')])
fig2 =go.Figure([go.Scatter(x=df_monthly.index, y = df_monthly['vr'],name = 'Volatility Volume Ratio')])


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(fig1.data[0], secondary_y=False)
fig.add_trace(fig2.data[0], secondary_y=True)

                 
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
    y=[50000,10000,1000],
    text=['Third Halving',
         'Second Halving',
          'First Halving'],
    mode="text",
))



fig.update_layout(xaxis = dict(title = 'Year'),
                  yaxis=dict(title = 'Price USD',type="log", range=[np.log10(1), np.log10(100000)]), 
                  yaxis2=dict(title = 'Volume',type="log", range=[np.log10(1), np.log10(10000)]))

py.iplot(fig)


# In[3]:


#focusing on the OBV value with price over 1 halving cycle

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np


fig1 =go.Figure([go.Scatter(x=df_weekly['weighted_price'].loc[df_weekly['cycle']==2020].index, y = df_weekly['weighted_price'].loc[df_weekly['cycle']==2020],name = 'BTC Price')])
fig2 =go.Figure([go.Scatter(x=df_weekly['Obv'].loc[df_weekly['cycle']==2020].index, y = df_weekly['Obv'].loc[df_weekly['cycle']==2020],name = 'On Balance Volume')])


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(fig1.data[0], secondary_y=False)
fig.add_trace(fig2.data[0], secondary_y=True)


fig.update_layout(xaxis = dict(title = 'Year'),
                  yaxis=dict(title = 'Price USD'), 
                  yaxis2=dict(title = 'OBV'))

py.iplot(fig)


# In[ ]:





# In[4]:


#checking correlation between the obv and the price. 
correlation_frame =pd.DataFrame()
correlation_frame['2020 Cycle'] = pd.Series(df_monthly['vr'].loc[df_monthly['cycle']==2020].values)
correlation_frame['2012 Cycle'] = pd.Series(df_monthly['vr'].loc[df_monthly['cycle']==2012].values)
correlation_frame['2016 Cycle'] = pd.Series(df_monthly['vr'].loc[df_monthly['cycle']==2016].values)
correlation_frame['2024 Cycle'] = pd.Series(df_monthly['vr'].loc[df_monthly['cycle']==2024].values)
correlation_frame = correlation_frame[['2012 Cycle','2016 Cycle','2020 Cycle','2024 Cycle']]

correlation_frame.corr(method='pearson')
#correlation_frame


# In[5]:


#plotting the price over each halving cycle and the OBV value over each halving cycle.

fig = go.Figure()

fig = make_subplots(rows=2, cols=4,subplot_titles=("Cycle 2009-2012", "Cycle 2012-2016","Cycle 2016-2020", "Cycle 2020-2024"))

fig.add_trace(go.Scatter(x=df_monthly['weighted_price'].loc[df_monthly['cycle']==2012].index, y = df_monthly['weighted_price'].loc[df_monthly['cycle']==2012],name = '2012 price'),row=1,col=1)
fig.add_trace(go.Scatter(x=df_monthly['weighted_price'].loc[df_monthly['cycle']==2016].index, y = df_monthly['weighted_price'].loc[df_monthly['cycle']==2016],name = '2016 price'),row=1,col=2)
fig.add_trace(go.Scatter(x=df_monthly['weighted_price'].loc[df_monthly['cycle']==2020].index, y = df_monthly['weighted_price'].loc[df_monthly['cycle']==2020],name = '2020 price'),row=1,col=3)
fig.add_trace(go.Scatter(x=df_monthly['weighted_price'].loc[df_monthly['cycle']==2024].index, y = df_monthly['weighted_price'].loc[df_monthly['cycle']==2024],name = '2024 price'),row=1,col=4)


fig.add_trace(go.Scatter(x=df_monthly['Obv'].loc[df_monthly['cycle']==2012].index, y = df_monthly['Obv'].loc[df_monthly['cycle']==2012],name = '2012 Obv'),row=2,col=1)
fig.add_trace(go.Scatter(x=df_monthly['Obv'].loc[df_monthly['cycle']==2016].index, y = df_monthly['Obv'].loc[df_monthly['cycle']==2016],name = '2016 Obv'),row=2,col=2)
fig.add_trace(go.Scatter(x=df_monthly['Obv'].loc[df_monthly['cycle']==2020].index, y = df_monthly['Obv'].loc[df_monthly['cycle']==2020],name = '2020 Obv'),row=2,col=3)
fig.add_trace(go.Scatter(x=df_monthly['Obv'].loc[df_monthly['cycle']==2024].index, y = df_monthly['Obv'].loc[df_monthly['cycle']==2024],name = '2024 Obv'),row=2,col=4)


# Update yaxis properties
fig.update_yaxes(title_text="Price", tickmode = 'auto',row=1, col=1)
fig.update_yaxes(tickmode = 'auto', row=2, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="OBV", tickmode = 'auto',row=2, col=1)
fig.update_yaxes(tickmode = 'auto', row=2, col=1)


fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

fig.show()


# In[6]:


correlation_frame =pd.DataFrame()
correlation_frame['2020 Cycle'] = pd.Series(df_monthly['Obv'].loc[df_monthly['cycle']==2020].values)
correlation_frame['2012 Cycle'] = pd.Series(df_monthly['Obv'].loc[df_monthly['cycle']==2012].values)
correlation_frame['2016 Cycle'] = pd.Series(df_monthly['Obv'].loc[df_monthly['cycle']==2016].values)
correlation_frame['2024 Cycle'] = pd.Series(df_monthly['Obv'].loc[df_monthly['cycle']==2024].values)
correlation_frame = correlation_frame[['2012 Cycle','2016 Cycle','2020 Cycle','2024 Cycle']]

correlation_frame.corr(method='pearson')
#correlation_frame


# In[7]:


# correlation matrix between the price at each halving cycle and the OBV at each halving cycle

correlation_frame =pd.DataFrame()
correlation_frame['2020 Cycle OBV'] = pd.Series(df_monthly['Obv'].loc[df_monthly['cycle']==2020].values)
correlation_frame['2012 Cycle OBV'] = pd.Series(df_monthly['Obv'].loc[df_monthly['cycle']==2012].values)
correlation_frame['2016 Cycle OBV'] = pd.Series(df_monthly['Obv'].loc[df_monthly['cycle']==2016].values)
correlation_frame['2024 Cycle OBV'] = pd.Series(df_monthly['Obv'].loc[df_monthly['cycle']==2024].values)
correlation_frame['2020 Cycle price'] = pd.Series(df_monthly['weighted_price'].loc[df_monthly['cycle']==2020].values)
correlation_frame['2012 Cycle price'] = pd.Series(df_monthly['weighted_price'].loc[df_monthly['cycle']==2012].values)
correlation_frame['2016 Cycle price'] = pd.Series(df_monthly['weighted_price'].loc[df_monthly['cycle']==2016].values)
correlation_frame['2024 Cycle price'] = pd.Series(df_monthly['weighted_price'].loc[df_monthly['cycle']==2024].values)
correlation_frame = correlation_frame[['2012 Cycle OBV','2016 Cycle OBV','2020 Cycle OBV','2024 Cycle OBV','2012 Cycle price','2016 Cycle price','2020 Cycle price','2024 Cycle price']]

correlation_frame.corr(method='pearson')
#correlation_frame


# In[8]:


correlation_frame = pd.DataFrame()
correlation_frame['OBV'] = df_daily['Obv']
correlation_frame['Price'] = df_daily['weighted_price']

correlation_frame.corr(method='pearson')


# In[9]:


correlation_frame = pd.DataFrame()
correlation_frame['OBV'] = df_weekly['Obv']
correlation_frame['Price'] = df_weekly['weighted_price']

correlation_frame.corr(method='pearson')


# In[11]:


correlation_frame = pd.DataFrame()
correlation_frame['OBV'] = df_monthly['Obv']
correlation_frame['Price'] = df_monthly['weighted_price']

correlation_frame.corr(method='pearson')


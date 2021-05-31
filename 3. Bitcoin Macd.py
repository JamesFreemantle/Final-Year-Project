#!/usr/bin/env python
# coding: utf-8

# In[1]:


#reading in datasets

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


#Zoomin in on one year only
df3 = df_daily.loc[(df_daily.index < pd.to_datetime('2021-12-31')) & (df_daily.index > pd.to_datetime('2021-01-01'))]


# In[3]:


df_monthly


# In[4]:


import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np

#since this is just for visualisation purposes to explain how macd works,
# the last 50 rows are not included in the graph because it makes the chart look messy.
fig1 =go.Figure([go.Scatter(x=df3.index, y = df3['weighted_price'][:-50],name = 'BTC Price')])
fig2 =go.Figure([go.Scatter(x=df3.index, y = df3['macd'][:-50],name = 'MACD')])
fig3 =go.Figure([go.Scatter(x=df3.index, y = df3['macds'][:-50],name = 'MACD SIGNAL')])


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(fig1.data[0], secondary_y=False)
fig.add_trace(fig2.data[0], secondary_y=True)
fig.add_trace(fig3.data[0], secondary_y=True)


# the overlaps between the signal line and th macd line is represented by either a red or green line depending on the direction of overlap.
for i in range(1,len(df3['macds'][:-50])):   
    macd = df3['macd']
    signal = df3['macds']
    
    if macd[i] > signal[i] and macd[i - 1] <= signal[i - 1]:
        fig.add_shape(type="line",
        x0=df3.index[i], y0=20000, x1=df3.index[i], y1=70000,
        line=dict(color="Green",width=3,dash = 'dot'))
    elif macd[i] < signal[i] and macd[i - 1] >= signal[i - 1]:
        fig.add_shape(type="line",
        x0=df3.index[i], y0=20000, x1=df3.index[i], y1=70000,
        line=dict(color="Red",width=3,dash = 'dot'))


fig.update_layout(xaxis = dict(title = 'Year'),
                  yaxis=dict(title = 'Price USD'), 
                  yaxis2=dict(title = 'Volume'))

py.iplot(fig)


# In[5]:


#for the period of the start of 2021 to present, what is the end capital following macd trading signals. 

capital = 10000
btc_bought = 0
for i in range(1,len(df3['macds'])):
    macd = df3['macd']
    signal = df3['macds']
    price = df3['weighted_price'][i]
    
    if macd[i] > signal[i-1] and macd[i - 1] <= signal[i - 1]:
        btc_bought = capital/price
    elif macd[i] < signal[i] and macd[i - 1] >= signal[i - 1]:
        if btc_bought != 0:
            capital = btc_bought *price
            btc_bought = 0
capital


# In[6]:


#macd strategy yearly
macd_df = pd.DataFrame(columns=['Year','Start Capital','End Capital','Profit/Loss'])  
#start at year 2012
year =2012
# loop for each of the ten years
for i in range(10):
    #from the start of the year to the end of the year
    df3 = df_daily.loc[(df_daily.index < pd.to_datetime(str(year)+'-12-31')) & (df_daily.index > pd.to_datetime(str(year)+'-01-01'))]
    #starting capital for each year is 10,000
    capital = 10000
    btc_bought = 0
    for j in range(1,len(df3['macds'])): #for each day
        macd = df3['macd']
        signal = df3['macds']
        price = df3['weighted_price'][j]

        if macd[j] > signal[j-1] and macd[j - 1] <= signal[j - 1]:
             btc_bought = capital/price
        elif macd[j] < signal[j] and macd[j - 1] >= signal[j - 1]:
            # ensuring that I have bitcoin to sell.
            if btc_bought != 0:
                capital = btc_bought *price
                btc_bought = 0
    macd_df = macd_df.append({'Year':str(year),'Start Capital':10000, 'End Capital': round(capital),'Profit/Loss': round(capital - 10000)},ignore_index = True)
    year = year+1
print(macd_df)


# In[34]:


#this dca strategy method is not used in final report
#10,000 to invest every year, resetting amount invested at start of every year
dca_df = pd.DataFrame(columns=['Year','Start Capital','End Capital','Profit/Loss'])  
year =2012
for i in range(10):
        
    df3 = df_daily.loc[(df_daily.index < pd.to_datetime(str(year)+'-12-31')) & (df_daily.index > pd.to_datetime(str(year)+'-01-01'))]
    btc_bought = 0
    for j in range(len(df3['macds'])):
        price = df3['weighted_price'][j]
        btc_bought = btc_bought + 27.4/price
        #10,000/365 = 27.4 invested per day
    capital = btc_bought * df3['weighted_price'][-1] #end
    dca_df = dca_df.append({'Year':str(year),'Start Capital':10000, 'End Capital': round(capital),'Profit/Loss': round(capital - 10000)},ignore_index = True)

    year = year+1
print(dca_df)


# In[29]:


df3


# In[38]:


# compared to buy and hold strategy
buyhold_df = pd.DataFrame(columns=['Year','Start Capital','End Capital','Profit/Loss'])  
year =2012
capital =10000
for i in range(10):
    df3 = df_daily.loc[(df_daily.index < pd.to_datetime(str(year)+'-12-31')) & (df_daily.index > pd.to_datetime(str(year)+'-01-01'))]
    
    btc_bought = capital/df3['weighted_price'][0] #start
    capital = btc_bought * df3['weighted_price'][-1] #end
    
    buyhold_df = buyhold_df.append({'Year':str(year),'Start Capital':10000, 'End Capital': round(capital),'Profit/Loss': round(capital - 10000)},ignore_index = True)
    capital =10000
    
    year = year+1
print(buyhold_df)


# In[9]:


#plotting the profit/loss returns of this strategy over each year
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np

fig1 =go.Figure([go.Scatter(x=macd_df['Year'][:], y = macd_df['Profit/Loss'][:],name = 'MACD')])
fig2 =go.Figure([go.Scatter(x=buyhold_df['Year'][:], y = buyhold_df['Profit/Loss'][:],name = 'BUY & HOLD')])
#fig3 =go.Figure([go.Scatter(x=dca_df['Year'][:], y = dca_df['Profit/Loss'][:],name = 'DCA')])

fig=go.Figure()
fig.add_trace(fig1.data[0])
fig.add_trace(fig2.data[0])
#fig.add_trace(fig3.data[0])

fig.update_layout(xaxis = dict(title = 'Year'),
                  yaxis=dict(title = 'Profit USD'))

py.iplot(fig)


# In[10]:


import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np

fig1 =go.Figure([go.Scatter(x=macd_df['Year'][2:], y = macd_df['Profit/Loss'][2:],name = 'MACD')])
fig2 =go.Figure([go.Scatter(x=buyhold_df['Year'][2:], y = buyhold_df['Profit/Loss'][2:],name = 'BUY & HOLD')])
#fig3 =go.Figure([go.Scatter(x=dca_df['Year'][2:], y = dca_df['Profit/Loss'][2:],name = 'DCA')])

fig=go.Figure()
fig.add_trace(fig1.data[0])
fig.add_trace(fig2.data[0])
#fig.add_trace(fig3.data[0])

fig.update_layout(xaxis = dict(title = 'Year'),
                  yaxis=dict(title = 'Profit USD'))

py.iplot(fig)


# In[11]:


import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np

fig1 =go.Figure([go.Scatter(x=macd_df['Year'][6:], y = macd_df['Profit/Loss'][6:],name = 'MACD')])
fig2 =go.Figure([go.Scatter(x=buyhold_df['Year'][6:], y = buyhold_df['Profit/Loss'][6:],name = 'BUY & HOLD')])
#fig3 =go.Figure([go.Scatter(x=dca_df['Year'][6:], y = dca_df['Profit/Loss'][6:],name = 'DCA')])

fig=go.Figure()
fig.add_trace(fig1.data[0])
fig.add_trace(fig2.data[0])
#fig.add_trace(fig3.data[0])

fig.update_layout(xaxis = dict(title = 'Year'),
                  yaxis=dict(title = 'Profit USD'))

py.iplot(fig)


#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
#reading in the data
df = pd.read_csv('C:/College/Final Year Project/daily_cleaned.csv')
# setting the 'Date' column to be the index and setting its type to 'DateTimeIndex'
df.index=pd.DatetimeIndex(df['Date'])
# deleteing the now redundant 'Date' column
del df['Date']
df


# In[2]:


# import plotly library
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np

#plotting the price in Linear vs Log
fig1 =go.Figure([go.Scatter(x=df.index, y = df['weighted_price'],name = 'BTC Price_Log')])
fig2 =go.Figure([go.Scatter(x=df.index, y = df['weighted_price'],name = 'BTC_Price_Linear')])


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(fig1.data[0], secondary_y=False)
fig.add_trace(fig2.data[0], secondary_y=True)

##adding vertical lines to highlight the halving event
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
                  yaxis=dict(title = 'BTC Price_Log',type="log", range=[np.log10(5), np.log10(100000)]), 
                  yaxis2=dict(title = 'BTC_Price_Linear'))

py.iplot(fig)


# In[3]:


#correlation dataframe of the prices between halving events
correlation_frame =pd.DataFrame()

#I entered cycle 2020 first, this creates an index large enough for the other halvings.
correlation_frame['2020 Cycle'] = pd.Series(df['weighted_price'].loc[df['cycle']==2020].values)
correlation_frame['2012 Cycle'] = pd.Series(df['weighted_price'].loc[df['cycle']==2012].values)
correlation_frame['2016 Cycle'] = pd.Series(df['weighted_price'].loc[df['cycle']==2016].values)
correlation_frame['2024 Cycle'] = pd.Series(df['weighted_price'].loc[df['cycle']==2024].values)
correlation_frame = correlation_frame[['2012 Cycle','2016 Cycle','2020 Cycle','2024 Cycle']]


# In[4]:


correlation_frame


# In[5]:


#since these columns have different length, the correlation is calculated in the rows filled. 
# Example/ '2012 Cycle' only has 333 rows, so the first 333 rows of other cycles are used for correlation 
correlation_frame.corr(method='pearson')


# In[7]:


#plotting the cycle in different figures 

fig = go.Figure()

fig = make_subplots(rows=1, cols=4,subplot_titles=("Cycle 2009-2012", "Cycle 2012-2016", "Cycle 2016-2020", "Cycle 2020-2024"))

fig.add_trace(go.Scatter(x=df['weighted_price'].loc[df['cycle']==2012].index, y = df['weighted_price'].loc[df['cycle']==2012],name = '2012 price'),row=1,col=1)
fig.add_trace(go.Scatter(x=df['weighted_price'].loc[df['cycle']==2016].index, y = df['weighted_price'].loc[df['cycle']==2016],name = '2016 price'),row=1,col=2)
fig.add_trace(go.Scatter(x=df['weighted_price'].loc[df['cycle']==2020].index, y = df['weighted_price'].loc[df['cycle']==2020],name = '2020 price'),row=1,col=3)
fig.add_trace(go.Scatter(x=df['weighted_price'].loc[df['cycle']==2024].index, y = df['weighted_price'].loc[df['cycle']==2024],name = '2024 price'),row=1,col=4)

# Update yaxis properties
fig.update_yaxes(title_text="Price", tickmode = 'auto',row=1, col=1)
fig.update_yaxes(tickmode = 'auto', row=1, col=2)
fig.update_yaxes(tickmode = 'auto', row=2, col=1)
fig.update_yaxes(tickmode = 'auto', row=2, col=2)

# Update yaxis properties
fig.update_xaxes(title_text="Year", tickmode = 'auto',row=1, col=1)
fig.update_xaxes(tickmode = 'auto', row=1, col=2)
fig.update_xaxes(tickmode = 'auto', row=2, col=1)
fig.update_xaxes(tickmode = 'auto', row=2, col=2)

fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

fig.show()


# # Gold vs Bitcoin

# In[7]:


pip install yfinance


# In[9]:


import yfinance as yf
#downlaoding gold price data from yahoofinance
gold_df = yf.download('GLD','2019-01-01','2021-5-24',auto_adjust=True)
gold_df = gold_df[['Close']]
gold_df = gold_df.dropna()


# In[10]:


gold_df


# In[11]:


#comparing this to bitcoin over the same timeframe
fig1 =go.Figure([go.Scatter(x=gold_df.index, y = gold_df['Close'],name = 'Gold')])
fig2 =go.Figure([go.Scatter(x=df['2019-01-1':].index, y = df['weighted_price']['2019-01-1':],name = 'Bitcoin')])


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(fig1.data[0], secondary_y=False)
fig.add_trace(fig2.data[0], secondary_y=True)


#plotting vertical lines when microstartegy, telsa and square first bought bitcoin. 
fig.add_shape(type="line",
    x0='2020-08-21', y0=110, x1='2020-08-21', y1=200,
    line=dict(color="MediumPurple",width=3,dash = 'dot'))
              
fig.add_shape(type="line",
    x0='2020-10-08', y0=110, x1='2020-10-08', y1=200,
    line=dict(color="MediumPurple",width=3,dash = 'dot'))

fig.add_shape(type="line",
    x0='2021-02-08', y0=110, x1='2021-02-08', y1=200,
    line=dict(color="MediumPurple",width=3,dash = 'dot'))

fig.add_trace(go.Scatter(
    x=['2020-08-21', '2020-10-08', '2021-02-08'],
    y=[140,160,195],
    text=['Microstrategy',
         'Square',
          'Tesla'],
    mode="text",
))



fig.update_layout(xaxis = dict(title = '2019-Present'),
                  yaxis=dict(title = 'Gold Price (Oz)'), 
                  yaxis2=dict(title = 'Bitcoin price'))

py.iplot(fig)


# In[12]:


#a piece of code to calculate the total bitcoin in issuance between the halvings, will be used for visualisation

halving_2012 = pd.to_datetime('2012-11-28')
halving_2016 = pd.to_datetime('2016-07-09')
halving_2020 = pd.to_datetime('2020-05-11')

issued_supply_list = [(160000*50)] #Aproximately 160000 was the block height on 01-01-12 when my dataset started
total_supply =0
block_reward = 50

for i in df.index:
    total_supply = issued_supply_list[-1]
    if i < halving_2012:
        adjusted_block_reward =  block_reward
        
    elif (i < halving_2016) & (i > halving_2012):
        adjusted_block_reward =  block_reward/2
        
    elif (i < halving_2020) & (i > halving_2016):
        adjusted_block_reward = block_reward/4
        
    elif i > halving_2020:
        adjusted_block_reward = block_reward/8
        
    issued_supply_list.append(total_supply + adjusted_block_reward *144) #Approx 144 blocks mined per day - Approx One Block mined every 10mins


# In[13]:


# comparing the bitcoin price to the total supply in issuance

fig1 =go.Figure([go.Scatter(x=df.index, y = df['weighted_price'],name = 'BTC Price')])
fig2 =go.Figure([go.Scatter(x=df.index, y = issued_supply_list, name = 'BTC Supply Issued')])


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
    y=[70000,40000,20000],
    text=['Third Halving',
         'Second Halving',
          'First Halving'],
    mode="text",
))



fig.update_layout(xaxis = dict(title = 'Year'),
                  yaxis=dict(title = 'BTC Price',type="log", range=[np.log10(1), np.log10(100000)]), 
                  yaxis2=dict(title = 'BTC Supply Issued'))

py.iplot(fig)


# In[14]:


#this is to find the supply curve
years = [2012]   #start at year 2012

#I've set the loop to stop at 32 becuase the last bitcoin will be mined in 2140. Which is 32 cycles of 4 years away 
for i in range(32): 
    years.append(years[-1]+4)

divide_50 = [50]
for i in range(32):
    divide_50.append(round(divide_50[-1]/2,6))
supply_frame = pd.DataFrame(divide_50,years)
supply_frame.columns =['Block Issuance']


# In[15]:


# again_these are 'previous to' figures. 50btc per block was rewarded previous to 2012. 
supply_frame


# In[16]:


import plotly.graph_objs as go 
#plotting block issuance over time
data = [go.Bar(x = years,y = divide_50 , name = 'BTC Block Issuance')]
fig = go.Figure(data=data)

fig.update_layout(xaxis = dict(title = 'Halving Events'),
                  yaxis=dict(title = 'Block Issuance', tickmode = 'array', tickvals = divide_50[:5]))


fig.add_trace(go.Scatter(
    x=['2050','2050','2050'],
   y=[28,15,9],
    text=['25 new BTC per block',
         '12.5 new BTC per block',
          '6.25 new BTC per block'],
    mode="text",
))


py.iplot(fig)


# In[17]:


#This is for calculating supply based on blocks, however blocks were mined on average slightly quicker than 10mins each, so 
# in the code above it was simpler for visulaisation puposes to calculate total btc issued using date rather than block height

issued_supply_list = []
total_supply =0
block_reward = 50
starting_block_height = 210000

stop_loop_height = 2000000

for count in range(stop_loop_height):
    total_supply = total_supply + block_reward 
    issued_supply_list.append(total_supply)
    if count == starting_block_height:
        starting_block_height = starting_block_height+210000
        block_reward = block_reward/2
    
issued_supply_frame = pd.DataFrame(issued_supply_list, columns=['supply'])


# In[18]:


issued_supply_list


# In[19]:


#plotting total suppy vs issuance

import plotly.graph_objs as go 
#plotting first ten bars only (for visualisation purposes)
fig1 = go.Figure([go.Bar(x = years[:10],y = divide_50[:10] , name = 'BTC Block Issuance')])
fig2 =go.Figure([go.Scatter(x=issued_supply_frame.index, y = issued_supply_frame['supply'], name = 'BTC Supply Issued')])

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(fig1.data[0], secondary_y=False)
fig.add_trace(fig2.data[0], secondary_y=True)

fig.update_layout(xaxis = dict(title = 'Year',tickmode='array', tickvals= years[:9]),
                  yaxis=dict(title = 'Block Reward'),
                  yaxis2 = dict(title = 'Supply Issued'))

fig.update_layout(xaxis2= {'title':'Block Height','anchor': 'y', 'overlaying': 'x', 'side': 'top','tickmode':'array', 'tickvals' : [210000,420000,630000,840000,1050000,1260000,1470000,1680000]})
fig.data[1].update(xaxis='x2')



py.iplot(fig)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[18]:


# reading csv files
import pandas as pd
df_daily = pd.read_csv('C:/College/Final Year Project/daily_cleaned.csv')
# setting index to date
df_daily = df_daily.set_index('Date')
# changing index type to DateTimeIndex
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


# In[19]:


df_daily.tail(5)


# In[20]:


import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np

#comparing the price with monthly rsi_12
fig1 =go.Figure([go.Scatter(x=df_daily.index, y = df_daily['weighted_price'],name = 'BTC Price')])
fig2 =go.Figure([go.Scatter(x=df_monthly.index, y = df_monthly['rsi_12'],name = 'RSI_12')])


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(fig1.data[0], secondary_y=False)
fig.add_trace(fig2.data[0], secondary_y=True)

#vertical lines when a halving event occurs
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
                  yaxis=dict(title = 'BTC Price',type="log", range=[np.log10(5), np.log10(100000)]), 
                  yaxis2=dict(title = 'RSI_12'))

py.iplot(fig)


# # Rsi Monthly

# In[21]:


#plotting rsi_12 values side by side for each halving cycle

fig = go.Figure()

fig = make_subplots(rows=1, cols=4,subplot_titles=("Cycle 2009-2012", "Cycle 2012-2016", "Cycle 2016-2020", "Cycle 2020-2024"))

fig.add_trace(go.Scatter(x=df_monthly['rsi_12'].loc[df_monthly['cycle']==2012].index, y = df_monthly['rsi_12'].loc[df_monthly['cycle']==2012],name = '2012 rsi_12'),row=1,col=1)
fig.add_trace(go.Scatter(x=df_monthly['rsi_12'].loc[df_monthly['cycle']==2016].index, y = df_monthly['rsi_12'].loc[df_monthly['cycle']==2016],name = '2016 rsi_12'),row=1,col=2)
fig.add_trace(go.Scatter(x=df_monthly['rsi_12'].loc[df_monthly['cycle']==2020].index, y = df_monthly['rsi_12'].loc[df_monthly['cycle']==2020],name = '2020 rsi_12'),row=1,col=3)
fig.add_trace(go.Scatter(x=df_monthly['rsi_12'].loc[df_monthly['cycle']==2024].index, y = df_monthly['rsi_12'].loc[df_monthly['cycle']==2024],name = '2024 rsi_12'),row=1,col=4)

# Update yaxis properties
fig.update_yaxes(title_text="Price", tickmode = 'auto',row=1, col=1)
fig.update_yaxes(tickmode = 'auto', row=1, col=2)
fig.update_yaxes(tickmode = 'auto', row=2, col=1)
fig.update_yaxes(tickmode = 'auto', row=2, col=2)

# Update yaxis properties
fig.update_xaxes(title_text="Price", tickmode = 'auto',row=1, col=1)
fig.update_xaxes(tickmode = 'auto', row=1, col=2)
fig.update_xaxes(tickmode = 'auto', row=2, col=1)
fig.update_xaxes(tickmode = 'auto', row=2, col=2)

fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

fig.show()


# In[22]:


# checking correlation of halvings (monthly data points)

correlation_frame =pd.DataFrame()
correlation_frame['2020 Cycle RSI'] = pd.Series(df_monthly['rsi_12'].loc[df_monthly['cycle']==2020].values)
correlation_frame['2012 Cycle RSI'] = pd.Series(df_monthly['rsi_12'].loc[df_monthly['cycle']==2012].values)
correlation_frame['2016 Cycle RSI'] = pd.Series(df_monthly['rsi_12'].loc[df_monthly['cycle']==2016].values)
correlation_frame['2024 Cycle RSI'] = pd.Series(df_monthly['rsi_12'].loc[df_monthly['cycle']==2024].values)
correlation_frame['2020 Cycle price'] = pd.Series(df_monthly['weighted_price'].loc[df_monthly['cycle']==2020].values)
correlation_frame['2012 Cycle price'] = pd.Series(df_monthly['weighted_price'].loc[df_monthly['cycle']==2012].values)
correlation_frame['2016 Cycle price'] = pd.Series(df_monthly['weighted_price'].loc[df_monthly['cycle']==2016].values)
correlation_frame['2024 Cycle price'] = pd.Series(df_monthly['weighted_price'].loc[df_monthly['cycle']==2024].values)
correlation_frame = correlation_frame[['2012 Cycle RSI','2016 Cycle RSI','2020 Cycle RSI','2024 Cycle RSI']]
#correlation_frame = correlation_frame[['2012 Cycle RSI','2016 Cycle RSI','2020 Cycle RSI','2024 Cycle RSI','2012 Cycle price','2016 Cycle price','2020 Cycle price','2024 Cycle price']]

correlation_frame.corr(method='pearson')
#correlation_frame


# In[23]:


correlation_frame =pd.DataFrame()
correlation_frame['RSI'] = pd.Series(df_monthly['rsi_12'])
correlation_frame['Price'] = pd.Series(df_monthly['weighted_price'])
correlation_frame.corr(method='pearson')


# # RSI weekly

# In[24]:


import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np

#comparing the price with monthly rsi_12
fig1 =go.Figure([go.Scatter(x=df_daily.index, y = df_daily['weighted_price'],name = 'BTC Price')])
fig2 =go.Figure([go.Scatter(x=df_weekly.index, y = df_weekly['rsi_14'],name = 'RSI_14')])


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(fig1.data[0], secondary_y=False)
fig.add_trace(fig2.data[0], secondary_y=True)

#vertical lines when a halving event occurs
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
                  yaxis=dict(title = 'BTC Price',type="log", range=[np.log10(5), np.log10(100000)]), 
                  yaxis2=dict(title = 'RSI_12'))

py.iplot(fig)


# In[25]:


# plotting weekly rsi_14 values across halving cycles
import plotly.offline as py
import plotly.graph_objs as go
fig = go.Figure()

fig = make_subplots(rows=1, cols=4,subplot_titles=("Cycle 2009-2012", "Cycle 2012-2016", "Cycle 2016-2020", "Cycle 2020-2024"))

fig.add_trace(go.Scatter(x=df_weekly['rsi_14'].loc[df_weekly['cycle']==2012].index, y = df_weekly['rsi_14'].loc[df_weekly['cycle']==2012],name = '2012 rsi_14'),row=1,col=1)
fig.add_trace(go.Scatter(x=df_weekly['rsi_14'].loc[df_weekly['cycle']==2016].index, y = df_weekly['rsi_14'].loc[df_weekly['cycle']==2016],name = '2016 rsi_14'),row=1,col=2)
fig.add_trace(go.Scatter(x=df_weekly['rsi_14'].loc[df_weekly['cycle']==2020].index, y = df_weekly['rsi_14'].loc[df_weekly['cycle']==2020],name = '2020 rsi_14'),row=1,col=3)
fig.add_trace(go.Scatter(x=df_weekly['rsi_14'].loc[df_weekly['cycle']==2024].index, y = df_weekly['rsi_14'].loc[df_weekly['cycle']==2024],name = '2024 rsi_14'),row=1,col=4)

# Update yaxis properties
fig.update_yaxes(title_text="Price", tickmode = 'auto',row=1, col=1)
fig.update_yaxes(tickmode = 'auto', row=1, col=2)
fig.update_yaxes(tickmode = 'auto', row=2, col=1)
fig.update_yaxes(tickmode = 'auto', row=2, col=2)

# Update yaxis properties
fig.update_xaxes(title_text="Price", tickmode = 'auto',row=1, col=1)
fig.update_xaxes(tickmode = 'auto', row=1, col=2)
fig.update_xaxes(tickmode = 'auto', row=2, col=1)
fig.update_xaxes(tickmode = 'auto', row=2, col=2)

fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

fig.show()


# In[26]:


# correlation of rsi_14 across halvings (weekly values)
correlation_frame =pd.DataFrame()
correlation_frame['2020 Cycle'] = pd.Series(df_weekly['rsi_14'].loc[df_weekly['cycle']==2020].values)
correlation_frame['2012 Cycle'] = pd.Series(df_weekly['rsi_14'].loc[df_weekly['cycle']==2012].values)
correlation_frame['2016 Cycle'] = pd.Series(df_weekly['rsi_14'].loc[df_weekly['cycle']==2016].values)
correlation_frame['2024 Cycle'] = pd.Series(df_weekly['rsi_14'].loc[df_weekly['cycle']==2024].values)
correlation_frame = correlation_frame[['2012 Cycle','2016 Cycle','2020 Cycle','2024 Cycle']]

correlation_frame.corr(method='pearson')
#correlation_frame


# In[27]:


correlation_frame =pd.DataFrame()
correlation_frame['RSI'] = pd.Series(df_weekly['rsi_14'])
correlation_frame['Price'] = pd.Series(df_weekly['weighted_price'])
correlation_frame.corr(method='pearson')


# # RSI Weekly Strategy

# In[28]:


#How accurately can RSI predict price movements over weekly price
profit_df = pd.DataFrame(columns=['Year','Start Capital','End Capital','Profit/Loss']) 
#starting year
year =2012
# number of years
for i in range(10):
    #between the start of the year and end of the year
    df3 = df_weekly.loc[(df_weekly.index < pd.to_datetime(str(year)+'-12-31')) & (df_weekly.index > pd.to_datetime(str(year)+'-01-01'))]
    # starting capital
    capital = 10000
    # starting btc_bought
    btc_bought = 0
    # for each row
    for j in range(1,len(df3['rsi_14'])):
        rsi_14 = df3['rsi_14']
        price = df3['weighted_price'][j]
        # upper bound of 70, lower bound of 50 
        # if the rsi is above 70, sell the bitcoin for capital, if the rsi is below 50, convert the capital into bitcoin
        if rsi_14[j] > 70 and 70 <= rsi_14[j - 1]:
            # chack that I have bitcoin to sell
            if btc_bought != 0:
                capital = btc_bought *price
                btc_bought = 0
        elif rsi_14[j] < 50 and 50 >= rsi_14[j - 1]:
             btc_bought = capital/price
    profit_df = profit_df.append({'Year':str(year),'Start Capital':10000, 'End Capital': round(capital),'Profit/Loss': round(capital - 10000)},ignore_index = True)
    year = year+1
    
print(profit_df)


# # RSI Daily

# In[29]:


import plotly.offline as py
import plotly.graph_objs as go
fig = go.Figure()

fig = make_subplots(rows=1, cols=4,subplot_titles=("Cycle 2009-2012", "Cycle 2012-2016", "Cycle 2016-2020", "Cycle 2020-2024"))

fig.add_trace(go.Scatter(x=df_daily['rsi_14'].loc[df_daily['cycle']==2012].index, y = df_daily['rsi_14'].loc[df_daily['cycle']==2012],name = '2012 rsi_12'),row=1,col=1)
fig.add_trace(go.Scatter(x=df_daily['rsi_14'].loc[df_daily['cycle']==2016].index, y = df_daily['rsi_14'].loc[df_daily['cycle']==2016],name = '2016 rsi_12'),row=1,col=2)
fig.add_trace(go.Scatter(x=df_daily['rsi_14'].loc[df_daily['cycle']==2020].index, y = df_daily['rsi_14'].loc[df_daily['cycle']==2020],name = '2020 rsi_12'),row=1,col=3)
fig.add_trace(go.Scatter(x=df_daily['rsi_14'].loc[df_daily['cycle']==2024].index, y = df_daily['rsi_14'].loc[df_daily['cycle']==2024],name = '2024 rsi_12'),row=1,col=4)

# Update yaxis properties
fig.update_yaxes(title_text="Price", tickmode = 'auto',row=1, col=1)
fig.update_yaxes(tickmode = 'auto', row=1, col=2)
fig.update_yaxes(tickmode = 'auto', row=2, col=1)
fig.update_yaxes(tickmode = 'auto', row=2, col=2)

# Update yaxis properties
fig.update_xaxes(title_text="Price", tickmode = 'auto',row=1, col=1)
fig.update_xaxes(tickmode = 'auto', row=1, col=2)
fig.update_xaxes(tickmode = 'auto', row=2, col=1)
fig.update_xaxes(tickmode = 'auto', row=2, col=2)

fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

fig.show()


# In[30]:


correlation_frame =pd.DataFrame()
correlation_frame['2020 Cycle'] = pd.Series(df_daily['rsi_14'].loc[df_daily['cycle']==2020].values)
correlation_frame['2012 Cycle'] = pd.Series(df_daily['rsi_14'].loc[df_daily['cycle']==2012].values)
correlation_frame['2016 Cycle'] = pd.Series(df_daily['rsi_14'].loc[df_daily['cycle']==2016].values)
correlation_frame['2024 Cycle'] = pd.Series(df_daily['rsi_14'].loc[df_daily['cycle']==2024].values)
correlation_frame = correlation_frame[['2012 Cycle','2016 Cycle','2020 Cycle','2024 Cycle']]

correlation_frame.corr(method='pearson')
#correlation_frame


# In[53]:


#How accurately can RSI predict price movements over daily price
profit_df = pd.DataFrame(columns=['Year','Start Capital','End Capital','Profit/Loss'])  
year =2012
for i in range(10):
    df3 = df_daily.loc[(df_daily.index < pd.to_datetime(str(year)+'-12-31')) & (df_daily.index > pd.to_datetime(str(year)+'-01-01'))]
    #print(df3)
    capital = 10000
    btc_bought = 0
    for j in range(1,len(df3['rsi_14'])):
        rsi_14 = df3['rsi_14']
        price = df3['weighted_price'][j]

        if rsi_14[j] > 70 and 70 <= rsi_14[j - 1]:
            if btc_bought != 0:
                capital = btc_bought *price
                btc_bought = 0
        elif rsi_14[j] < 50 and 50 >= rsi_14[j - 1]:
             btc_bought = capital/price
    profit_df = profit_df.append({'Year':str(year),'Start Capital':10000, 'End Capital': round(capital),'Profit/Loss': round(capital - 10000)},ignore_index = True)
    year = year+1
print(profit_df)


# In[54]:


df_daily


# In[55]:


# compared to buy and hold strategy
profit_df = pd.DataFrame(columns=['Year','Start Capital','End Capital','Profit/Loss'])  
year =2012
capital =10000
for i in range(10):
    df3 = df_daily.loc[(df_daily.index < pd.to_datetime(str(year)+'-12-31')) & (df_daily.index > pd.to_datetime(str(year)+'-01-01'))]
    
    btc_bought = capital/df3['weighted_price'][0] #start
    capital = btc_bought * df3['weighted_price'][-1] #end
    
    profit_df = profit_df.append({'Year':str(year),'Start Capital':10000, 'End Capital': round(capital),'Profit/Loss': round(capital - 10000)},ignore_index = True)
    capital =10000
    
    year = year+1
print(profit_df)


# In[ ]:





# In[9]:


correlation_frame =pd.DataFrame()
correlation_frame['RSI'] = pd.Series(df_daily['rsi_14'])
correlation_frame['Price'] = pd.Series(df_daily['weighted_price'])
correlation_frame.corr(method='pearson')


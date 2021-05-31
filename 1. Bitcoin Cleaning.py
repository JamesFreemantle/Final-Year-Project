#!/usr/bin/env python
# coding: utf-8

# In[1]:


#bitstamp data found on kaggle @ https://www.kaggle.com/kognitron/zielaks-bitcoin-historical-data-wo-nan
import pandas as pd
df = pd.read_csv('C:/College/Final Year Project/bitstamp_cleaned.csv',sep=',')
df.head()


# In[2]:


len(df.index)
#number of rows in the data


# In[3]:


#This block of code takes 10 mins to run on my machine.
#sets the index to DateTime and changes tyoe to DateTimeIndex
df.set_index('DateTime',inplace = True)
df.index=pd.DatetimeIndex(df.index)


# In[4]:


print(df.index.dtype)
#check to make sure index of of tyoe datetime


# In[5]:


df


# In[6]:


#We want our data to have the Average prices, but the sum of volume durinf a period
mean= df.groupby(df.index.date).mean()
total = df.groupby(df.index.date).sum()

data = {'Open':mean['Open'],
        'High':mean['High'],
        'Low':mean['Low'],
        'Close':mean['Close'],
        'Volume_BTC':total['Volume_(BTC)'],
        'Volume':total['Volume_(Currency)'],
        'Weighted_Price':mean['Weighted_Price']}

df =pd.DataFrame(data,index = mean.index)
df


# In[13]:


#round to 4 decimal places.
df = df.round(decimals = 4)


# In[14]:


#data sourced from :     https://www.cryptodatadownload.com/data/bitstamp/
# download the BTC/USD daily file from this link. 
import pandas as pd
#read in the BTC/USD daily file
df2 = pd.read_csv('C:/College/Final Year Project/Bitstamp_BTCUSD_d.csv')
df2


# In[15]:


#check number of rows in data
len(df2.index)


# In[16]:


#reversing the order from newest to oldest -> oldest to newest
df2 = df2.iloc[::-1]


# In[17]:


#same process as previous dataset, detting index to DateTime
df2.set_index('date',inplace = True)
df2.index=pd.DatetimeIndex(df2.index)
df2.index.name = None


# In[18]:


df2


# In[19]:


#preparing to concatonate two datasets together by removing overlapping rows from df2
df2 = df2.loc['2020-09-15':]
del df2['unix']
del df2['symbol']


# In[20]:


#setting the weighted price column to the average between intraday high and low. 
df2['Weighted_Price'] = df2[['high','low']].mean(axis=1)
df2.rename(columns = {'open':'Open','high':'High','low':'Low','close':'Close',
                      'Volume BTC':'Volume_BTC','Volume USD':'Volume'}, inplace = True)


# In[21]:


#concatonate the two dataframes together and set index type to DateTimeIndex
frames = [df,df2]
new_df = pd.concat(frames,axis=0)
new_df.index=pd.DatetimeIndex(new_df.index)


# In[26]:


new_df


# In[25]:


#removing last row becuase that's today and data such as total volume is lower
new_df = new_df[:-1]


# In[27]:


#Writing this new df to a csv for accessibility reasons
new_df.to_csv('C:/College/Final Year Project/daily_bitstamp.csv')


# # Reading the preprocessed data

# In[28]:


#read in the data that has just been sent to csv. 
import pandas as pd
df = pd.read_csv('C:/College/Final Year Project/daily_bitstamp.csv')
#reading in the csv sets the index to a number, so I'm resetting it to a DateTimeIndex
df = df.set_index('Unnamed: 0')
df.index.rename('Date',inplace = True)
df.index=pd.DatetimeIndex(df.index)
df


# In[29]:


#These are the dates of the previous 3 bitcoin halvings
halving_2012 = pd.to_datetime('2012-11-28')
halving_2016 = pd.to_datetime('2016-07-09')
halving_2020 = pd.to_datetime('2020-05-11')

# dividing total dataframe into 4 for halving cycle analysis
#'pre' in this context is short for previous
pre_2012 = df.loc[df.index < halving_2012]
pre_2016 = df.loc[(df.index < halving_2016) & (df.index > halving_2012)]
pre_2020 = df.loc[(df.index < halving_2020) & (df.index > halving_2016)]
pre_2024 = df.loc[df.index > halving_2020]

#creating a column called 'Cycle' in each dataframe specifying what halving cycle they're in, will be useful when they are concatonated together
pre_2012['Cycle'] = '2012'
pre_2016['Cycle'] = '2016'
pre_2020['Cycle'] = '2020'
pre_2024['Cycle'] = '2024'

frames = [pre_2012,pre_2016,pre_2020,pre_2024]
new_df = pd.concat(frames,sort = False)


# In[ ]:





# In[30]:


#We want our data to have Average prices, but the sum of volume

#monthly figures
#transforming the dataframe so each row is grouped by the month 
monthly_mean= new_df.groupby([new_df.index.strftime('%Y-%m')]).mean()
total= new_df.groupby([new_df.index.strftime('%Y-%m')]).sum()

data = {'Open':monthly_mean['Open'],
        'High':monthly_mean['High'],
        'Low':monthly_mean['Low'],
        'Close':monthly_mean['Close'],
        'Volume_BTC':total['Volume_BTC'],
        'Volume':total['Volume'],
        'Weighted_Price':monthly_mean['Weighted_Price']}

monthly_mean =pd.DataFrame(data,index = monthly_mean.index)

#weekly figures
#transforming the dataframe so each row is grouped by the week
weekly_mean = new_df.groupby([new_df.index.strftime('%Y-%W')]).mean()
total= new_df.groupby([new_df.index.strftime('%Y-%W')]).sum()

data = {'Open':weekly_mean['Open'],
        'High':weekly_mean['High'],
        'Low':weekly_mean['Low'],
        'Close':weekly_mean['Close'],
        'Volume_BTC':total['Volume_BTC'],
        'Volume':total['Volume'],
        'Weighted_Price':weekly_mean['Weighted_Price'],}

weekly_mean =pd.DataFrame(data,index = weekly_mean.index)


# In[31]:


weekly_mean


# In[32]:


#removing the last row in the data since it could be an incomplete week.
weekly_mean.drop(weekly_mean.tail(1).index,inplace=True)


# In[33]:


# to change the groupby index to a datetime index, I am specifiying what day of the week the row of data is. In strptime, '0' = 1st day of week
weekly_mean.index = weekly_mean.index.astype(str)+'-0'


# In[34]:


weekly_mean.index


# In[35]:


from datetime import datetime

correct_date_list =[]

# '%w' is the symbol for the day (0-6) of the week 
for i in weekly_mean.index:    
    correct_date = datetime.strptime(i,"%Y-%W-%w")
    correct_date_list.append(correct_date)
    


# In[36]:


weekly_mean.index = correct_date_list


# In[37]:


print(weekly_mean.index.dtype)


# In[38]:


#data grouped by week with a datetime index
weekly_mean


# In[39]:


#repeating these steps with monthly figures
monthly_mean


# In[40]:


#removing the last row in the data since it could be an incomplete month.
monthly_mean.drop(monthly_mean.tail(1).index,inplace=True)


# In[41]:


#checking index type, need to change to DateTimeIndex
print(monthly_mean.index.dtype)


# In[42]:


# adding an '-01' for strptime to identify the datetime format
monthly_mean.index = monthly_mean.index.astype(str)+'-01'


# In[43]:


monthly_mean


# In[44]:


from datetime import datetime

correct_date_list =[]

for i in monthly_mean.index:    
    correct_date = datetime.strptime(i,"%Y-%m-%d")
    correct_date_list.append(correct_date)
    


# In[45]:


monthly_mean.index = correct_date_list


# In[46]:


print(monthly_mean.index.dtype)


# In[ ]:





# In[ ]:





# In[47]:


#within monthly dataframes and weekly dataframes I am adding a column specifying which halving cycle the row belongs to

#These are the dates of the previous 3 bitcoin halvings
halving_2012 = pd.to_datetime('2012-11-28')
halving_2016 = pd.to_datetime('2016-07-09')
halving_2020 = pd.to_datetime('2020-05-11')

pre_2012_weekly = weekly_mean.loc[weekly_mean.index < halving_2012]
pre_2016_weekly = weekly_mean.loc[(weekly_mean.index < halving_2016) & (weekly_mean.index > halving_2012)]
pre_2020_weekly = weekly_mean.loc[(weekly_mean.index < halving_2020) & (weekly_mean.index > halving_2016)]
pre_2024_weekly = weekly_mean.loc[weekly_mean.index > halving_2020]

pre_2012_monthly = monthly_mean.loc[monthly_mean.index < halving_2012]
pre_2016_monthly = monthly_mean.loc[(monthly_mean.index < halving_2016) & (monthly_mean.index > halving_2012)]
pre_2020_monthly = monthly_mean.loc[(monthly_mean.index < halving_2020) & (monthly_mean.index > halving_2016)]
pre_2024_monthly = monthly_mean.loc[monthly_mean.index > halving_2020]

pre_2012_weekly['Cycle'] = '2012'
pre_2016_weekly['Cycle'] = '2016'
pre_2020_weekly['Cycle'] = '2020'
pre_2024_weekly['Cycle'] = '2024'

pre_2012_monthly['Cycle'] = '2012'
pre_2016_monthly['Cycle'] = '2016'
pre_2020_monthly['Cycle'] = '2020'
pre_2024_monthly['Cycle'] = '2024'

#This is to add new column called cycle to overall dataframe
weekly_frames = [pre_2012_weekly,pre_2016_weekly,pre_2020_weekly,pre_2024_weekly]
weekly_mean = pd.concat(weekly_frames,sort = False)

monthly_frames = [pre_2012_monthly,pre_2016_monthly,pre_2020_monthly,pre_2024_monthly]
monthly_mean = pd.concat(monthly_frames,sort = False)


# In[48]:


weekly_mean


# In[49]:


#to install stockstats, whic will generate the indicators. 
pip install stockstats


# In[50]:


#adding the indicators from stockstats library. 
from stockstats import StockDataFrame as Sdf
monthly_mean = Sdf.retype(monthly_mean)
monthly_mean.get('rsi_12')
monthly_mean.get('vr')
monthly_mean.get('macd')

weekly_mean = Sdf.retype(weekly_mean)
weekly_mean.get('rsi_14')
weekly_mean.get('vr')
weekly_mean.get('macd')

new_df = Sdf.retype(new_df)
new_df.get('rsi_14')
new_df.get('vr')
new_df.get('macd')


# In[51]:


#adding OBV manually because stockstats does not support OBV.
# The OBV is calulated using the 'Close' price and volume. 
import numpy as np
monthly_obv = (np.sign(monthly_mean['close'].diff()) * monthly_mean['volume']).fillna(0).cumsum().to_frame()
weekly_obv = (np.sign(weekly_mean['close'].diff()) * weekly_mean['volume']).fillna(0).cumsum().to_frame()
daily_obv = (np.sign(new_df['close'].diff()) * new_df['volume']).fillna(0).cumsum().to_frame()

#renaming the column of the dataframe
monthly_obv = monthly_obv.rename(columns = {0:'obv'})
weekly_obv = weekly_obv.rename(columns = {0:'obv'})
daily_obv = daily_obv.rename(columns = {0:'obv'})

#since the dataframe 'monthly_obv' only has one column, it can be added as a column to another datframe as if it's a series
monthly_mean['Obv']=monthly_obv
weekly_mean['Obv']=weekly_obv
new_df['Obv']=daily_obv


# In[52]:


monthly_mean


# In[53]:


weekly_mean


# In[54]:


new_df


# In[ ]:





# In[55]:


#writing these datframes with indicators to csv files so they can be used for analysis in another notebook. 
new_df.to_csv('C:/College/Final Year Project/daily_cleaned.csv')


# In[56]:


monthly_mean.to_csv('C:/College/Final Year Project/monthly_cleaned.csv')


# In[57]:


weekly_mean.to_csv('C:/College/Final Year Project/weekly_cleaned.csv')


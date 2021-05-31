#!/usr/bin/env python
# coding: utf-8

# In[144]:


# importing libraries
import numpy as np
import pandas as pd

import plotly.graph_objs as go 

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils.vis_utils import plot_model


# # MLP (Neural Net) Traditional indicators

# In[145]:


# a method to calculate the Mean_Absolute_Percentage_Error (MAPE)

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


# In[146]:


# setting up an MLPRegressor function
def myMLPRegressor(x_train,y_train,x_test,y_test):
    # Initialising standard scaler
    scaler = StandardScaler()
    # Fitting the scaler with x_train data
    scaler.fit(x_train)
    #scaling x_train and y_train data
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    #creating the model:
        # setting random_state to 1: This ensures the same result is produced across calls, by determining the same weight and bias initialisation
        # 10000 maximum number of iterations
        # 100 is the default hidden layer size. (number of neurons in hidden layer)
        # setting activation function to identity. returns returns f(x) = x
        # setting learning rate to adaptive, this means that the learning rate is constant if the training loss decreases. 
    MLP = MLPRegressor(random_state=1, max_iter=10000, hidden_layer_sizes = (100), activation = 'identity',learning_rate = 'adaptive').fit(x_train_scaled, y_train)
    # use the model to precit the price based on the scaled x_test data
    MLP_pred = MLP.predict(x_test_scaled)
    
    #creating a dataframe for binary classification (up or down)
    classification = pd.DataFrame(data=y_test,columns =['test'])
    #if the price in price data increases from previous day, input 1. Else input 0
    classification['test'] = np.where(classification['test'].shift(-1) > classification['test'], 1, 0)
    classification['pred'] = MLP_pred
    classification['pred'] = np.where(classification['pred'].shift(-1) > classification['pred'], 1, 0)
    
    MLP_acc = accuracy_score(classification['test'],classification['pred'])
    
    # calculate the mean squared error
    MLP_MSE = mean_squared_error(y_test, MLP_pred)
    
    MLP_MAPE =mean_absolute_percentage_error(y_test, MLP_pred)
    # calculate the r2 score
    MLP_R2 = r2_score(MLP_pred, y_test)
    
    return MLP_R2 , MLP_MSE, MLP_pred, MLP_MAPE, MLP_acc


# In[147]:


def runMLP(pre_2012,pre_2016,pre_2020,pre_2024,indicator_set):
# list of frames, will be used to iterate through them
    frames = [pre_2012,pre_2016,pre_2020,pre_2024]
    # will be used to print out which cycle the data belongs to
    frames_string = ['2009-2012','2012-2016','2016-2020','2020-2024']
    MLP_results = pd.DataFrame({'Cycle':[],'R2 score':[],'MSE_score':[]})

    count = 0
    for df in frames:

        shift = -10
        df['Price_lag'] = df['Price'].shift(shift)

        train_pt = int(len(df)*.8)
        train = df.iloc[:train_pt,:]
        test = df.iloc[train_pt:,:]

        # x training on indicators only
        x_train = train.iloc[:shift,1:-1]
        # y training on what the indicators should predict after 10 days
        y_train = train['Price_lag'][:shift]
        x_test = test.iloc[:shift,1:-1]
        y_test = test['Price'][:shift]
        # changing y_test to an array, output of MLP_pred is array, so to compare MLP_pred to y_test, y_test needs to be an array
        y_test = np.array(y_test)

        # calling the MLPRegressor method and supplying inputs
        MLP_R2, MLP_MSE, MLP_pred, MLP_MAPE, MLP_acc = myMLPRegressor(x_train,y_train,x_test,y_test)
        
        #creating a row of the cycles scores
        new_row = pd.Series(data={'Cycle':frames_string[count],'R2 score':MLP_R2,'MSE_score':MLP_MSE, 'MAPE':MLP_MAPE, 'classification':MLP_acc})

        # appending the results to the dataframe
        MLP_results = MLP_results.append(new_row,ignore_index = True)

        #plotting the prection vs the actual
        fig1 = go.Scatter(x=train.index,y=train['Price'],name = 'Train Actual')
        fig2 = go.Scatter(x=test.index[:shift],y=test['Price'],name = 'Test Actual')
        fig3 = go.Scatter(x=test.index[:shift],y=MLP_pred,name = 'Prediction') 

        line = {'data': [fig1,fig2,fig3],
              'layout': {
                  'xaxis' :{'title': 'Date'},
                  'yaxis' :{'title': '$'},
                  'title' : 'MLP' + ' - ' + str(frames_string[count]) + ' - ' + indicator_set
              }}
        fig = go.Figure(line)
        fig.show()

        count = count+1

    return MLP_results


# In[148]:


# Importing Training Set
#reading in csvs
df_daily = pd.read_csv('C:/College/Final Year Project/daily_cleaned.csv')
#setting index
df_daily = df_daily.set_index('Date')
# changing index type to DateTimeIndex
df_daily.index=pd.DatetimeIndex(df_daily.index)
# removing first 4 rows because indicator values are 100 or NaN (not accurate)
df_daily = df_daily.iloc[4:]

data = {'Price':df_daily['weighted_price'],
        'Macd':df_daily['macd'],
        'Macd_signal': df_daily['macds'],
        'RSI':df_daily['rsi_14'],
        'Obv':df_daily['Obv']}

df_daily =pd.DataFrame(data,index = df_daily.index)

halving_2012 = pd.to_datetime('2012-11-28')
halving_2016 = pd.to_datetime('2016-07-09')
halving_2020 = pd.to_datetime('2020-05-11')

#spitting into 4 halving cycle dataframe
pre_2012 = df_daily.loc[df_daily.index < halving_2012]
pre_2016 = df_daily.loc[(df_daily.index < halving_2016) & (df_daily.index > halving_2012)]
pre_2020 = df_daily.loc[(df_daily.index < halving_2020) & (df_daily.index > halving_2016)]
pre_2024 = df_daily.loc[(df_daily.index > halving_2020)]


# In[149]:


indicator_set = 'Traditional Indicators'
MLP_results = runMLP(pre_2012,pre_2016,pre_2020,pre_2024,indicator_set)
print(MLP_results)


# # MLP (Neural Net) On-Chain Indicators

# In[135]:


#repeating for On-Chain indicators
df_daily = pd.read_csv('C:/College/Final Year Project/new_indicators.csv')
df_daily = df_daily.set_index('date')
df_daily.index=pd.DatetimeIndex(df_daily.index)

data = {'MVRV' : df_daily['CapMVRVCur'],
    'NVT' : df_daily['NVTAdj90'],
    'Fee' : df_daily['FeeTotNtv'],
    'Price' : df_daily['PriceUSD']}

df_daily =pd.DataFrame(data,index = df_daily.index)

halving_2012 = pd.to_datetime('2012-11-28')
halving_2016 = pd.to_datetime('2016-07-09')
halving_2020 = pd.to_datetime('2020-05-11')

pre_2012 = df_daily.loc[df_daily.index < halving_2012]
pre_2016 = df_daily.loc[(df_daily.index < halving_2016) & (df_daily.index > halving_2012)]
pre_2020 = df_daily.loc[(df_daily.index < halving_2020) & (df_daily.index > halving_2016)]
pre_2024 = df_daily.loc[(df_daily.index > halving_2020)]


# In[136]:


#calling the same method as Traditional Indicators for completely fair comaprison between the two
indicator_set = 'On-Chain Indicators'
MLP_results = runMLP(pre_2012,pre_2016,pre_2020,pre_2024,indicator_set)
print(MLP_results)


# In[ ]:





# # LSTM (Neural Net) Traditional Indicators

# In[137]:


# cross validation technique which is a variation of k-fold
timesplit= TimeSeriesSplit(n_splits=4)


# In[138]:


#Defining the setup of the LSTM model
def LSTM_method(trainX,X_train1,y_train,X_test1,y_test):    
    
    
    #LSTM mode: The number of units represnts the dimensionality
    #           The actiavtion function is set to 'relu' (smae as MLPRegressor)
    #           Return sequences set to true for returning the last output in output.
    lstm = Sequential()
    
    lstm.add(LSTM(units = 64, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=True))
    lstm.add(LSTM(units = 32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=True))
    lstm.add(LSTM(units = 16, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
    
    lstm.add(Dense(1))
    # the error is computed using MSE
    # Adam optimization is a stochastic gradient descent method
    lstm.compile(loss='mean_squared_error', optimizer='adam')
    
    # fitting the model on x_train and y_train over 50 epochs with batch size 5
    # verbose =1 shows the training progress pe epoch, setting to 0 hides the progress
    lstm.fit(X_train1, y_train, epochs=50, batch_size=5, verbose=1, shuffle=False)
    
    #predicting the price based on the test indicator data
    y_pred= lstm.predict(X_test1)
    #return a copy of the predicyion flattened into 1 dimension
    y_pred = y_pred.flatten()
    
    classification = pd.DataFrame(data=y_test,columns =['test'])
    classification['test'] = np.where(classification['test'].shift(-1) > classification['test'], 1, 0)
    classification['pred'] = y_pred
    classification['pred'] = np.where(classification['pred'].shift(-1) > classification['pred'], 1, 0)
    
    LSTM_acc = accuracy_score(classification['test'],classification['pred'])
    
    #calculate the MSE between the y_test and prediction from model
    LSTM_MSE = mean_squared_error(y_test, y_pred)
    MAPE = mean_absolute_percentage_error(y_test,y_pred)
    LSTM_r2 = r2_score(y_test,y_pred)
    
    return y_pred, LSTM_MSE, MAPE, LSTM_r2, LSTM_acc


# In[139]:


def runLSTM(features, pre_2012, pre_2016, pre_2020, pre_2024,indicator_set):
    
    # creating a list of frames to iterate through
    frames = [pre_2012,pre_2016,pre_2020,pre_2024]
    frames_string = ['2009-2012','2012-2016','2016-2020','2020-2024']
    #creating a dataframe for cycle, mean-squared error and r2 score.
    LSTM_results = pd.DataFrame({'Cycle':[],'MSE_score':[],'R2_score':[]})
    count = 0

    for df in frames:
        # initialise the scaler
        scaler = StandardScaler()
        #scale the indicators
        feature_transform = scaler.fit_transform(df[features])
        #create a dataframe of the scaled data
        feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)
        # the desired output variable is price
        output_var = pd.DataFrame(df['Price'])
        
        #timesplit divides the data into folds
        for train_index, test_index in timesplit.split(feature_transform):
            
            #performing a train test split foe each fold in the timesplit
            X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]

            y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()
        # transforming to numpy array
        trainX =np.array(X_train)
        testX =np.array(X_test)
        
        #shaping the data to be accepted by the model
        X_trainfit = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_testfit = testX.reshape(X_test.shape[0], 1, X_test.shape[1])    
        
        #running the LSTM model with parameters
        LSTM_pred, LSTM_MSE,MAPE, LSTM_r2, LSTM_acc = LSTM_method(trainX,X_trainfit,y_train,X_testfit,y_test)
        
        #creating a row from model output
        new_row = pd.Series(data={'Cycle':frames_string[count],'MSE_score':LSTM_MSE,'MAPE':MAPE, 'classification':LSTM_acc})

        #append the results to a dataframe
        LSTM_results = LSTM_results.append(new_row,ignore_index = True)
        
        
        #methods to calaculate the accuraccy, f1 score and ROC curve score
        #print(' Accuracy: { :0.3 }'.format( 100*accuracy_score(y_test, 1 * (LSTM_pred > 0.5))) )

        #print(' f1 score: {:0.3}'.format( 100*f1_score( y_test , 1 * (LSTM_pred > 0.5))))

        #print(' ROC AUC: {:0.3}'.format( roc_auc_score( y_test , LSTM_pred)) )

        #plotting the prediction vs actual
        fig1 = go.Scatter(x=X_train.index,y=output_var['Price'],name = 'Train Actual') # Training actuals
        fig2 = go.Scatter(x=X_test.index,y=y_test,name = 'Test Actual') # Testing actuals
        fig3 = go.Scatter(x=X_test.index,y=LSTM_pred,name = 'Prediction') # Testing predction

        # Combine in an object  
        line = {'data': [fig1,fig2,fig3],
              'layout': {
                  'xaxis' :{'title': 'Date'},
                  'yaxis' :{'title': '$'},
                  'title' : 'LSTM' + ' - ' + frames_string[count]+ ' - ' + indicator_set
              }}
        # Send object to a figure 
        fig = go.Figure(line)

        # Show figure
        fig.show()

        count = count+1

    return LSTM_results


# In[140]:


# Importing Training Set
df_daily = pd.read_csv('C:/College/Final Year Project/daily_cleaned.csv')
df_daily = df_daily.set_index('Date')
df_daily.index=pd.DatetimeIndex(df_daily.index)
df_daily = df_daily.iloc[4:]

data = {'Price':df_daily['weighted_price'],
        'Macd':df_daily['macd'],
        'Macds':df_daily['macds'],
        'RSI':df_daily['rsi_14'],
        'Obv':df_daily['Obv']}

df_daily =pd.DataFrame(data,index = df_daily.index)

halving_2012 = pd.to_datetime('2012-11-28')
halving_2016 = pd.to_datetime('2016-07-09')
halving_2020 = pd.to_datetime('2020-05-11')

pre_2012 = df_daily.loc[df_daily.index < halving_2012]
pre_2016 = df_daily.loc[(df_daily.index < halving_2016) & (df_daily.index > halving_2012)]
pre_2020 = df_daily.loc[(df_daily.index < halving_2020) & (df_daily.index > halving_2016)]
pre_2024 = df_daily.loc[(df_daily.index > halving_2020)]


# In[141]:


#specifying the indicator names
features = ['Macd','Macds','RSI','Obv']
indicator_set = 'Traditional Indicators'
#passed into the model with halving cycle data
LSTM_results = runLSTM(features, pre_2012,pre_2016,pre_2020,pre_2024, indicator_set)
#output results dataframe
print(LSTM_results)


# # LSTM (Neural Net) On-Chain Indicators

# In[142]:


#repeating with O-chain indicators
df_daily = pd.read_csv('C:/College/Final Year Project/new_indicators.csv')
df_daily = df_daily.set_index('date')
df_daily.index=pd.DatetimeIndex(df_daily.index)

data = {'MVRV' : df_daily['CapMVRVCur'],
    'NVT' : df_daily['NVTAdj90'],
    'Fee' : df_daily['FeeTotNtv'],
    'Price' : df_daily['PriceUSD']}

df_daily =pd.DataFrame(data,index = df_daily.index)

halving_2012 = pd.to_datetime('2012-11-28')
halving_2016 = pd.to_datetime('2016-07-09')
halving_2020 = pd.to_datetime('2020-05-11')

pre_2012 = df_daily.loc[df_daily.index < halving_2012]
pre_2016 = df_daily.loc[(df_daily.index < halving_2016) & (df_daily.index > halving_2012)]
pre_2020 = df_daily.loc[(df_daily.index < halving_2020) & (df_daily.index > halving_2016)]
pre_2024 = df_daily.loc[(df_daily.index > halving_2020)]


# In[143]:


indicator_set = 'On-Chain Indicators'
features = ['MVRV','NVT','Fee']
LSTM_results = runLSTM(features, pre_2012,pre_2016,pre_2020,pre_2024,indicator_set)
print(LSTM_results)


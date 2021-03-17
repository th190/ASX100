import sys
import numpy as np
import torch
import random

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_model_inputs(stock, data, n_back, n_forward):
    x_train = np.array(data[stock])
    
    new_data = []
    for i in range(0, x_train.shape[0] - n_back - n_forward - 1):
        new_data.append(x_train[i: i + n_back + n_forward])
    
    x_train = np.array(new_data, dtype='float32')  
    ## X [batch_size, n_back, n_features]
    ## y [batch_size, n_features]
    x = x_train[:,:n_back,:]
    y = x_train[:,n_back:,3]

    return x, y
        

def load_stock_data(full_data, n_back, test_date_split="2019-01-01", TI=True):
    import talib
    stock_date = full_data.sort_values(['Code', 'Date']).groupby('Code').head(1)
    stock_date = stock_date[stock_date['Date'] == '2015-01-02']
    
    all_stocks = stock_date['Code']

    full_data = full_data[full_data['Code'].isin(all_stocks)]

    stock_data = {}
    for stock in all_stocks:
        if not TI:
            stock_data[stock] = full_data.loc[full_data['Code'] == stock, ['Date', 'Close']].set_index('Date').sort_index()
        else:
            stock_data[stock] = full_data.loc[full_data['Code'] == stock, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].set_index('Date').sort_index()
    

    for stock in all_stocks:
        if not TI:
            break
        stock_data[stock]['SMA'] = talib.SMA(stock_data[stock]['Close'],timeperiod=5) #Simple Moving Average 
        stock_data[stock]['ATR'] = talib.ATR(stock_data[stock]['High'], stock_data[stock]['Low'], stock_data[stock]['Close'],timeperiod=5) #Average True Range
        stock_data[stock]['ADX'] = talib.ADX(stock_data[stock]['High'], stock_data[stock]['Low'], stock_data[stock]['Close'], timeperiod=5) #Average Directional Movement Index
        stock_data[stock]['slowk'], stock_data[stock]['slowd'] = talib.STOCH(stock_data[stock]['High'], stock_data[stock]['Low'], stock_data[stock]['Close'],
                                                                            fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

        stock_data[stock] = stock_data[stock].dropna()   

    train_data = {}
    test_data = {}
    for stock in all_stocks:
        train_data[stock] = stock_data[stock][stock_data[stock].index < test_date_split]
        test_data[stock] = stock_data[stock][stock_data[stock].index >= test_date_split]

    scalers = {stock: StandardScaler() for stock in all_stocks}
    for stock in all_stocks:
        train_data[stock][:] = scalers[stock].fit_transform(train_data[stock])
        test_data[stock][:] = scalers[stock].transform(test_data[stock])
        test_data[stock] = pd.concat([train_data[stock].iloc[-n_back:], test_data[stock]], axis=0)
    
    return train_data, test_data, scalers




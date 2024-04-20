import datetime
import time

from bitoproClient.bitopro_indicator import indicator
from bitoproClient.bitopro_restful_client import BitoproRestfulClient, CandlestickResolution, OrderType, StatusKind
from bitoproClient.bitopro_util import get_current_timestamp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

import uicredential as ui
# Here we use pandas and matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pickle

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
# from keras.inputs import Inputs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
# from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
# from keras.layers import Dense
from keras.layers import LSTM, Dropout, Dense
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
import tensorflow as tf

from backtesting import Backtest, Strategy

cwd = os.getcwd()


def get_hist_yahoo():
    ticker = 'SOL-USD'
    start_dt = '2020-01-01'
    end_dt = '2024-04-01'
    data_frame = yf.download(tickers=ticker, start=start_dt, prepost=True, progress=False)
    data_frame = data_frame.drop(columns='Adj Close')
    return data_frame


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg# convert series to supervised learning
def series_to_supervised_2(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    # df = DataFrame(data)
    df = data
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def preprocess_1(df):
    """
    Create the target variable
    """
    # print(df.head(3))
    df['Change'] = df['Close'].pct_change(-1)
    df.Change = df.Change * -1
    df.Change = np.where(df.Change > 0, 1, 0)
    df.insert(0, 'Change', df.pop('Change'))
    # df.Close = df.Close * 100
    # df.Close = df.Close *
    df_cpy = df.dropna().copy()
    # change_tmr = np.where(df_cpy.Change > 0, 1, 0)
    # df_cpy.pop('Change')
    # data_here.insert(0, 'Close', data_here.pop('Close'))
    # df_cpy.insert(loc=0, column='Change', value=change_tmr)
    # del df_cpy['Close']
    # Remove rows with any missing data
    # df = df.dropna().copy()
    # a = df.shape[0]
    # df.drop([a - 1, a - 2, a - 3])
    # df.drop([0, 1])
    print(df_cpy.head(10))
    # print(df.index)
    # print(df.loc[['2020-04-10']])
    return df_cpy
def preprocess_2(df):
    """
    Create the target variable
    """
    # print(df.head(3))
    # df['Close'] = df['Close'].pct_change(-1)
    # df.Close = df.Close * -1
    # df.Close = df.Close * 100
    # df.Close = df.Close *
    df_cpy = df.dropna().copy()
    # change_tmr = np.where(df_cpy.Close > 0, 1, 0)
    # df_cpy.insert(loc=0, column='Change', value=change_tmr)
    # del df_cpy['Close']
    # Remove rows with any missing data
    # df = df.dropna().copy()
    # a = df.shape[0]
    # df.drop([a - 1, a - 2, a - 3])
    # df.drop([0, 1])
    print(df_cpy.head(10))
    # print(df.index)
    # print(df.loc[['2020-04-10']])
    return df_cpy


def train_class():
    data_here = get_hist_yahoo()
    df_explanatory = data_here

    df_explanatory = preprocess_2(data_here)

    # data_here.insert(0, 'Close', data_here.pop('Close'))
    # data_here.insert(0, 'Close', data_here.pop('Close'))
    data_here = preprocess_1(data_here)
    df_explanatory = data_here
    data_here_val = data_here.values
    # ensure all data is float
    data_here_val = data_here_val.astype('float32')
    num_feat = 6

    # frame as supervised learning
    look_back = 2
    data_here_val = series_to_supervised(data_here_val, look_back, 1)
    df_explanatory_2 = series_to_supervised_2(df_explanatory, look_back, 1)

    # return
    col_drop_list = []
    for col_drop in range(num_feat * look_back + 1, num_feat * (look_back + 1), 1):
        if col_drop % num_feat != 0:
            col_drop_list.append(col_drop)

    data_here_val.drop(data_here_val.columns[col_drop_list],
                       axis=1,
                       inplace=True)
    df_explanatory_2.drop(df_explanatory_2.columns[col_drop_list],
                       axis=1,
                       inplace=True)

    # print(df_explanatory_2.head(5))
    # print(data_here_val.head(5))
    # return
    # results = df_explanatory_2.loc[df_explanatory_2["column_name"] == my_value]
    # my_list = [1, 2, 3]
    # print(data_here_val)
    # print("data_here_val")
    print(df_explanatory)
    print("df_explanatory")
    my_list = df_explanatory_2.index
    # my_list = data_here_val[data_here_val.columns[-1]]
    # print(len(my_list))
    # print("my_list")
    results = df_explanatory.loc[my_list]
    df_explanatory_3 = pd.concat([results, df_explanatory_2], axis=1)
    # print(df_explanatory_3)
    # return

    # results = df_explanatory.loc[df_explanatory["Change"].isin(my_list)]
    #
    # print(df_explanatory.index)
    print(results)
    print("results")
    # print(data_here_val[data_here_val.columns[-1]])
    # return
    # df_explanatory = data_here_val[data_here_val.columns[:-1]]
    # df_explanatory = data_here_val.iloc[[:, 1]]
    # data_proc = data_here_val.values.astype('float32')
    data_proc = df_explanatory_2.values.astype('float32')
    x = data_proc[:, :-1]
    scaler_min_max_x = MinMaxScaler(feature_range=(0, 1))
    # print(x)
    x = scaler_min_max_x.fit_transform(x)
    train_ratio = .7
    n_sample = x.shape[0]
    train_size = int(n_sample * train_ratio)
    test_size = n_sample - train_size
    x = x.reshape((n_sample, look_back, num_feat))
    y = data_proc[:, -1]
    y = y.reshape((-1, 1))

    x_train, y_train = x[:train_size, :], y[:train_size, :]
    x_test, y_test = x[train_size:, :], y[train_size:, :]
    # x0 = x[4, :, :]
    # x0 = x0.reshape((1, look_back, num_feat))
    # y0 = y[4]

    model = Sequential()
    model.add(LSTM(50, activation=keras.activations.relu,
                   input_shape=(look_back, num_feat),
                   return_sequences=True))
    model.add(Dropout(.2))
    model.add(LSTM(10, return_sequences=True,
                   activation=keras.activations.relu))
    model.add(Dropout(.4))
    model.add(LSTM(2, activation=keras.activations.relu))
    model.add(Dropout(.1))
    model.add(Dense(1, activation=keras.activations.sigmoid))

    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.002, epsilon=1e-07),
                  metrics=[keras.metrics.BinaryAccuracy(threshold=0.5),
                           keras.metrics.Recall(thresholds=0.5),
                           keras.metrics.Precision(thresholds=0.5),
                           keras.metrics.F1Score(threshold=0.5)])
    # fit model
    # model.fit(x_train, y_train, batch_size=20, epochs=5, validation_split=.3, verbose=0)
    history = model.fit(x, y, batch_size=20, epochs=40, validation_split=.3, verbose=0)

    # evaluate model
    # score = model.evaluate(x_test, y_test, batch_size=20, verbose=1)
    # y0_pred = model.predict(x0, verbose=0)
    y_pred = model.predict(x_test, verbose=0)
    y_test = y_test.reshape(len(array(y_test)), 1)
    y_pred = y_pred.reshape(len(array(y_pred)), 1)
    y_pred = 1 * np.array(tf.greater(y_pred, .5))
    y_result = np.concatenate((y_pred, y_test), axis=1)

    train_save_num = 1
    file_name_save_model = f'{cwd}/resource/models/model_class_lstm_{train_save_num}.keras'
    model.save(file_name_save_model)

    file_name_save_scaler_min_max = f'{cwd}/resource/scalers/scaler_class_min_max_{train_save_num}.pkl'

    with open(file_name_save_scaler_min_max, 'wb') as f:
        pickle.dump(scaler_min_max_x, f)

    # print('Test score:', score)
    # print(y_result)
    f_score_test = array(history.history['val_f1_score'])
    f_score_train = array(history.history['f1_score'])

    loss_test = array(history.history['val_loss'])
    loss_train = array(history.history['loss'])

    loss_test = array(history.history['val_loss'])
    loss_train = array(history.history['loss'])

    binary_accuracy_test = array(history.history['val_binary_accuracy'])
    binary_accuracy_train = array(history.history['binary_accuracy'])

    precision_test = array(history.history['val_precision'])
    precision_train = array(history.history['precision'])

    recall_test = array(history.history['val_recall'])
    recall_train = array(history.history['recall'])

    # plot history
    pyplot.plot(precision_train, label='train')
    pyplot.plot(precision_test, label='test')
    pyplot.legend()
    pyplot.show()

    class SimpleClassificationUD(Strategy):
        def __init__(self, broker, data, params):
            super().__init__(broker, data, params)
            self.already_bought = None
            self.model = None

        def init(self):
            self.model = model
            self.scaler = scaler_min_max_x
            self.already_bought = False

        def next(self):
            # print(self.data.df.size)
            look_back = 2
            num_feat = 6
            # # data_here = self.data.df
            # data_here = self.data
            # print(data_here)
            # data_here.insert(0, 'Close', data_here.pop('Close'))
            # data_here = preprocess_1(data_here)
            # data_here_val = data_here.values
            # # ensure all data is float
            # data_here_val = data_here_val.astype('float32')
            # num_feat = 5
            #
            # # frame as supervised learning
            # look_back = 2
            # data_here_val = series_to_supervised(data_here_val, look_back, 1)
            # col_drop_list = []
            # for col_drop in range(num_feat * look_back + 1, num_feat * (look_back + 1), 1):
            #     if col_drop % num_feat != 0:
            #         col_drop_list.append(col_drop)
            #
            # data_here_val.drop(data_here_val.columns[col_drop_list],
            #                    axis=1,
            #                    inplace=True)
            df_explanatory = self.data.df.values[-1, 6:-1]
            # # df_explanatory = data_here_val.iloc[[:, 1]]
            # # data_proc = data_here_val.values.asty.tail(1)
            # data_proc = data_here_val
            # data_proc = data_proc.values.astype('float32')
            # x = data_proc[-1, :-1]
            # scaler_min_max_x = MinMaxScaler(feature_range=(0, 1))
            # # print(x)
            # # x = scaler_min_max_x.fit_transform(x)
            df_explanatory = self.scaler.transform(df_explanatory.reshape(1, len(df_explanatory)))
            # # train_ratio = .7
            n_sample = df_explanatory.shape[0]
            #
            explanatory_today = df_explanatory.reshape((n_sample, look_back, num_feat))
            #
            forecast_tomorrow = self.model.predict(explanatory_today, verbose=0)
            forecast_tomorrow = forecast_tomorrow.reshape(len(array(forecast_tomorrow)), 1)
            forecast_tomorrow = forecast_tomorrow.reshape(len(array(forecast_tomorrow)), 1)
            forecast_tomorrow = 1 * np.array(tf.greater(forecast_tomorrow, .5))
            forecast_tomorrow = forecast_tomorrow[0]
            # # explanatory_today = self.data.df.iloc[-1:, :]
            # forecast_tomorrow = self.model.predict(explanatory_today)[0]
            # forecast_tomorrow = 1

            # conditions to sell or buy
            if forecast_tomorrow == 1 and self.already_bought == False:
                self.buy()
                self.already_bought = True
            elif forecast_tomorrow == 0 and self.already_bought == True:
                self.sell()
                self.already_bought = False
            else:
                pass

    # df_explanatory = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    bt = Backtest(df_explanatory_3, SimpleClassificationUD,
                  cash=10000, commission=.002, exclusive_orders=True)
    results = bt.run()
    print(results.to_frame(name='Values').loc[:'Return [%]'])


def predict_class(x0):
    train_save_num = 1
    file_name_save_model = f'{cwd}/resource/models/model_class_lstm_{train_save_num}.keras'
    # model.save(file_name_save_model)
    model = load_model(file_name_save_model)

    file_name_save_scaler_min_max = f'{cwd}/resource/scalers/scaler_class_min_max_{train_save_num}.pkl'

    with open(file_name_save_scaler_min_max, 'rb') as f:
        scaler_min_max_x = pickle.load(f)

    x0 = scaler_min_max_x.transform(x0.reshape(1, len(x0)))

    look_back = 2
    num_feat = 5
    x0 = x0.reshape((1, look_back, num_feat))
    y0 = model.predict(x0, verbose=0)
    y0 = y0.reshape(len(array(y0)), 1)
    y0 = y0.reshape(len(array(y0)), 1)
    y0 = 1 * np.array(tf.greater(y0, .5))
    # y_result = np.concatenate((y_pred, y_test), axis=1)
    return y0


def get_x0_y0(idx):
    data_here = get_hist_yahoo()
    data_here.insert(0, 'Close', data_here.pop('Close'))
    data_here = preprocess_1(data_here)
    data_here_val = data_here.values
    # ensure all data is float
    data_here_val = data_here_val.astype('float32')
    num_feat = 5

    # frame as supervised learning
    look_back = 2
    data_here_val = series_to_supervised(data_here_val, look_back, 1)
    col_drop_list = []
    for col_drop in range(num_feat * look_back + 1, num_feat * (look_back + 1), 1):
        if col_drop % num_feat != 0:
            col_drop_list.append(col_drop)

    data_here_val.drop(data_here_val.columns[col_drop_list],
                       axis=1,
                       inplace=True)
    data_proc = data_here_val.values.astype('float32')
    return data_proc[idx, :]
    # x = data_proc[:, :-1]


def main():
    train_class()
    # idx = 30
    # x_data = get_x0_y0(idx)
    # x0 = x_data[:-1]
    # y0 = x_data[-1]
    # y0_pred = predict_class(x0)
    # # y0_result = np.concatenate((y0_pred, y0), axis=1)
    # # y0_result = np.concatenate((y0_pred, y0), axis=1)
    # # print(y0_result)
    # print(y0_pred)
    # print(y0)


if __name__ == "__main__":
    main()

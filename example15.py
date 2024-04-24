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

import datetime
# from datetime import date

from backtesting import Backtest, Strategy

cwd = os.getcwd()


def get_hist_yahoo(ticker='SOL-USD', start_dt='2020-01-01', end_dt=None):
    # ticker = 'SOL-USD'
    # start_dt = '2020-01-01'
    # end_dt = '2024-04-01'
    if end_dt is None:
        end_dt = str(datetime.date.today())

    data_frame = yf.download(tickers=ticker, start=start_dt, end=end_dt, prepost=True, progress=False)
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
    return agg  # convert series to supervised learning


def series_to_supervised_2(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
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


def map_past_timestamp_to_target(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
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


def append_binary_target(df):
    """
    Create the target variable
    """
    df['Change'] = df['Close'].pct_change(-1)
    df.Change = df.Change * -1
    df.Change = np.where(df.Change > 0, 1, 0)
    df.insert(0, 'Change', df.pop('Change'))
    df_cpy = df.dropna().copy()

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


class SimpleClassificationUD(Strategy):
    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.model_train_save_num = None
        self.model_file = None
        self.scaler = None
        self.already_bought = None
        self.model = None

    def init(self):
        self.model_train_save_num = 1
        self.model_file = f'{cwd}/resource/models/model_class_lstm_{self.model_train_save_num}.keras'
        self.model = load_model(self.model_file)
        save_scaler_min_max_file = f'{cwd}/resource/scalers/scaler_class_min_max_{self.model_train_save_num}.pkl'

        with open(save_scaler_min_max_file, 'rb') as f:
            self.scaler = pickle.load(f)
        self.already_bought = False

    def next(self):
        y_t = .4
        num_next_days = 1
        look_back = 2
        num_feat = 6
        df_explanatory = self.data.df.values[-1, 6:-1]
        df_explanatory = self.scaler.transform(df_explanatory.reshape(1, len(df_explanatory)))
        n_sample = df_explanatory.shape[0]
        explanatory_today = df_explanatory.reshape((n_sample, look_back, num_feat))
        forecast_tomorrow = self.model.predict(explanatory_today, verbose=0)
        forecast_tomorrow = forecast_tomorrow.reshape(len(array(forecast_tomorrow)), 1)
        forecast_tomorrow = forecast_tomorrow.reshape(len(array(forecast_tomorrow)), 1)
        forecast_tomorrow = 1 * np.array(tf.greater(forecast_tomorrow, y_t))
        print(forecast_tomorrow)
        forecast_tomorrow = forecast_tomorrow.flatten()
        forecast_tomorrow = forecast_tomorrow[0]
        print(forecast_tomorrow)

        # conditions to sell or buy
        if forecast_tomorrow == 1 and self.already_bought is False:
            self.buy()
            self.already_bought = True
        elif forecast_tomorrow == 0 and self.already_bought is True:
            self.sell()
            self.already_bought = False
        else:
            pass


def train_class():
    y_t = .4
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
    model.add(Dropout(.1))
    # model.add(LSTM(5, return_sequences=True,
    #                activation=keras.activations.relu))
    # model.add(Dropout(.4))
    model.add(LSTM(2, activation=keras.activations.relu))
    model.add(Dropout(.2))
    model.add(Dense(1, activation=keras.activations.sigmoid))

    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07),
                  metrics=[keras.metrics.BinaryAccuracy(threshold=y_t),
                           keras.metrics.Recall(thresholds=y_t),
                           keras.metrics.Precision(thresholds=y_t),
                           keras.metrics.F1Score(threshold=y_t)])
    # fit model
    # model.fit(x_train, y_train, batch_size=20, epochs=5, validation_split=.3, verbose=0)
    history = model.fit(x, y, batch_size=20, epochs=200, validation_split=.3, verbose=0)

    # evaluate model
    # score = model.evaluate(x_test, y_test, batch_size=20, verbose=1)
    # y0_pred = model.predict(x0, verbose=0)
    y_pred = model.predict(x_test, verbose=0)
    y_test = y_test.reshape(len(array(y_test)), 1)
    y_pred = y_pred.reshape(len(array(y_pred)), 1)
    y_pred = 1 * np.array(tf.greater(y_pred, y_t))
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
    pyplot.plot(precision_train, label='train p')
    pyplot.plot(precision_test, label='test p')

    pyplot.plot(recall_train, label='train r')
    pyplot.plot(recall_test, label='test r')

    pyplot.plot(f_score_train, label='train f')
    pyplot.plot(f_score_test, label='test f')
    pyplot.legend()
    pyplot.show()

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


def extract_data(ticker='SOL-USD', start_dtt='2020-01-01', end_dtt=None, train_save_num=1):
    if end_dtt is None:
        end_dtt = str(datetime.date.today())
    data_used = get_hist_yahoo(ticker='SOL-USD', start_dt=start_dtt, end_dt=end_dtt)

    start_dtt = datetime.datetime.strptime(start_dtt, '%Y-%m-%d').strftime('%y_%m_%d')
    end_dtt = datetime.datetime.strptime(end_dtt, '%Y-%m-%d').strftime('%y_%m_%d')

    ticker = ticker.replace("-", "_").lower()
    file_data = f'{cwd}/resource/data/raw/data_raw_ti_{ticker}_st_{start_dtt}_ed_{end_dtt}_n_{train_save_num}.csv'
    data_used.to_csv(file_data)

    return data_used


def load_data(ticker='SOL-USD', start_dtt='2020-01-01', end_dtt=None, train_save_num=1):
    if end_dtt is None:
        end_dtt = str(datetime.date.today())

    start_dtt = datetime.datetime.strptime(start_dtt, '%Y-%m-%d').strftime('%y_%m_%d')
    end_dtt = datetime.datetime.strptime(end_dtt, '%Y-%m-%d').strftime('%y_%m_%d')
    ticker = ticker.replace("-", "_").lower()
    file_data = f'{cwd}/resource/data/raw/data_raw_ti_{ticker}_st_{start_dtt}_ed_{end_dtt}_n_{train_save_num}.csv'
    data_retrieved = pd.read_csv(file_data, parse_dates=['Date'], index_col=0)
    return data_retrieved


def load_data_extended(ticker='SOL-USD', start_dtt='2020-01-01', end_dtt=None, train_save_num=1):
    if end_dtt is None:
        end_dtt = str(datetime.date.today())

    start_dtt = datetime.datetime.strptime(start_dtt, '%Y-%m-%d').strftime('%y_%m_%d')
    end_dtt = datetime.datetime.strptime(end_dtt, '%Y-%m-%d').strftime('%y_%m_%d')
    ticker = ticker.replace("-", "_").lower()
    file_data = f'{cwd}/resource/data/processed/data_proc_ti_{ticker}_st_{start_dtt}_ed_{end_dtt}_n_{train_save_num}.csv'
    data_retrieved = pd.read_csv(file_data, parse_dates=['Date'], index_col=0)
    return data_retrieved


def process_data(extracted_data, n_pst_dys=1, n_nxt_dys=1, ticker='SOL-USD', start_dtt='2020-01-01', end_dtt=None,
                 train_save_num=1):
    data_with_target = append_binary_target(extracted_data)
    num_feat = data_with_target.shape[1]
    data_with_target_extended = map_past_timestamp_to_target(data_with_target, n_pst_dys, n_nxt_dys)
    col_drop_list = []
    for col_drop in range(num_feat * n_pst_dys + n_nxt_dys, num_feat * (n_pst_dys + n_nxt_dys), 1):
        if col_drop % num_feat != 0:
            col_drop_list.append(col_drop)

    data_with_target_extended.drop(data_with_target_extended.columns[col_drop_list],
                                   axis=1,
                                   inplace=True)

    df_train = data_with_target.loc[data_with_target_extended.index]
    df_backtest = pd.concat([df_train, data_with_target_extended], axis=1)

    if end_dtt is None:
        end_dtt = str(datetime.date.today())
    start_dtt = datetime.datetime.strptime(start_dtt, '%Y-%m-%d').strftime('%y_%m_%d')
    end_dtt = datetime.datetime.strptime(end_dtt, '%Y-%m-%d').strftime('%y_%m_%d')

    ticker = ticker.replace("-", "_").lower()
    file_data = f'{cwd}/resource/data/processed/data_proc_ti_{ticker}_st_{start_dtt}_ed_{end_dtt}_n_{train_save_num}.csv'
    df_backtest.to_csv(file_data)

    return df_backtest


def preprocess_x(data_x, train_save_num=1):
    scaler_min_max_x = MinMaxScaler(feature_range=(0, 1))
    data_x_out = scaler_min_max_x.fit_transform(data_x)
    file_name_save_scaler_min_max = f'{cwd}/resource/scalers/scaler_class_min_max_{train_save_num}.pkl'
    with open(file_name_save_scaler_min_max, 'wb') as f:
        pickle.dump(scaler_min_max_x, f)
    return data_x_out


def train_data(data_train, test_ratio=.3, y_t=.4, num_feat=6, n_pst_dys=1, n_nxt_dys=1,
               ticker='SOL-USD', start_dtt='2020-01-01', end_dtt=None, train_save_num=1):
    x = data_train.iloc[:, num_feat:-n_nxt_dys].values.astype('float32')
    x_prep = preprocess_x(x)

    n_sample = x.shape[0]
    n_timestamp_with_feat = x.shape[1]
    x_prep = x_prep.reshape((n_sample, n_pst_dys, num_feat))
    # fix next time  for multivariate prediction
    y = data_train.iloc[:, -n_nxt_dys:].values.astype('float32')
    # as uni-variate is not 2d data transform required for lstm
    if n_nxt_dys == 1:
        y = y.reshape((-1, 1))

    model = Sequential()
    model.add(LSTM(50, activation=keras.activations.relu,
                   input_shape=(n_pst_dys, num_feat),
                   return_sequences=True))
    model.add(Dropout(.1))
    model.add(LSTM(2, activation=keras.activations.relu))
    model.add(Dropout(.2))
    model.add(Dense(1, activation=keras.activations.sigmoid))

    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07),
                  metrics=[keras.metrics.BinaryAccuracy(threshold=y_t),
                           keras.metrics.Recall(thresholds=y_t),
                           keras.metrics.Precision(thresholds=y_t),
                           keras.metrics.F1Score(threshold=y_t)])

    history = model.fit(x_prep, y, batch_size=20, epochs=200, validation_split=test_ratio, verbose=1)

    train_save_num = 1
    file_name_save_model = f'{cwd}/resource/models/model_class_lstm_{train_save_num}.keras'
    model.save(file_name_save_model)

    if end_dtt is None:
        end_dtt = str(datetime.date.today())

    start_dtt = datetime.datetime.strptime(start_dtt, '%Y-%m-%d').strftime('%y_%m_%d')
    end_dtt = datetime.datetime.strptime(end_dtt, '%Y-%m-%d').strftime('%y_%m_%d')
    ticker = ticker.replace("-", "_").lower()

    # file_history = f'{cwd}/resource/reports/histories/{0}/hist_ti_{ticker}_st_{start_dtt}_ed_{end_dtt}_n_{
    # train_save_num}{1} '
    file_history = '{0}/resource/reports/histories/train/{1}/hist_train_ti_{2}_st_{3}_ed_{4}_n_{5}{6} '
    pd.DataFrame(history.history).to_csv(
        file_history.format(cwd, 'csv', ticker, start_dtt, end_dtt, train_save_num, '.csv'))
    with open(file_history.format(cwd, 'pkl', ticker, start_dtt, end_dtt, train_save_num, '.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    return history.history


def back_test(data_test, money=10000, ticker='SOL-USD', start_dtt='2020-01-01', end_dtt=None, train_save_num=1):
    test_engine = Backtest(data_test, SimpleClassificationUD,
                           cash=money, commission=.002, exclusive_orders=True)
    test_history = test_engine.run()
    test_history = test_history.to_frame(name='Values').loc[:'Return [%]']
    test_history = test_history.to_frame(name='Values').loc[:'Return [%]']
    if end_dtt is None:
        end_dtt = str(datetime.date.today())

    start_dtt = datetime.datetime.strptime(start_dtt, '%Y-%m-%d').strftime('%y_%m_%d')
    end_dtt = datetime.datetime.strptime(end_dtt, '%Y-%m-%d').strftime('%y_%m_%d')

    ticker = ticker.replace("-", "_").lower()
    file_hist = f'{cwd}/resource/reports/histories/test/data_hist_test_ti_{ticker}_st_{start_dtt}_ed_{end_dtt}_n_{train_save_num}.csv'
    test_history.to_csv(file_hist)

    return test_history


def main():
    start_date = '2020-01-01'
    # extracted_data = extract_data(start_dtt=start_date)
    # loaded_data = load_data(start_dtt=start_date)
    # print(loaded_data)
    n_past_days = 2
    # data_extended = process_data(extracted_data, n_pst_dys=n_past_days)
    # data_extended = process_data(loaded_data, n_pst_dys=n_past_days)
    loaded_data_extended = load_data_extended(start_dtt=start_date)
    # print(loaded_data_extended)
    # train_history = train_data(data_extended, n_pst_dys=n_past_days)
    # train_history = train_data(loaded_data_extended, n_pst_dys=n_past_days)
    invest_money = 10000
    # test_history = back_test(data_extended, money=invest_money)
    test_history = back_test(loaded_data_extended, money=invest_money)
    # print(test_history)
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

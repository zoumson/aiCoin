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
# from keras.inputs import Input
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
import re

cwd = os.getcwd()
global_model_train_save_num = 1
global_next_dys = 1
global_past_dys = 3
global_y_threshold = .4
global_num_feature = 6


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
        self.y_t = None
        self.n_nxt_dys = None
        self.num_feat = None
        self.n_pst_dys = None
        self.model_train_save_num = None
        self.model_file = None
        self.scaler = None
        self.already_bought = None
        self.model = None

    def init(self):
        self.model_train_save_num = global_model_train_save_num
        self.y_t = global_y_threshold
        self.n_nxt_dys = global_next_dys
        self.n_pst_dys = global_past_dys
        self.num_feat = global_num_feature
        self.model_file = f'{cwd}/resource/models/compiled/model_class_lstm_{self.model_train_save_num}.keras'
        self.model = load_model(self.model_file)
        save_scaler_min_max_file = f'{cwd}/resource/scalers/scaler_class_min_max_{self.model_train_save_num}.pkl'

        with open(save_scaler_min_max_file, 'rb') as f:
            self.scaler = pickle.load(f)
        self.already_bought = False

    def next(self):

        df_explanatory = self.data.df.iloc[-1:, self.num_feat:-self.n_nxt_dys].values.astype('float32')
        df_explanatory = self.scaler.transform(df_explanatory)
        n_sample = df_explanatory.shape[0]
        explanatory_today = df_explanatory.reshape((n_sample, self.n_pst_dys, self.num_feat))
        forecast_tomorrow = self.model.predict(explanatory_today, verbose=0)
        forecast_tomorrow = 1 * np.array(tf.greater(forecast_tomorrow, self.y_t))
        forecast_tomorrow = forecast_tomorrow.flatten()

        for dy in range(self.n_nxt_dys):
            if forecast_tomorrow[dy] == 1 and self.already_bought is False:
                self.buy()
                self.already_bought = True
            elif forecast_tomorrow[dy] == 0 and self.already_bought is True:
                self.sell()
                self.already_bought = False
            else:
                pass


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
    for col_drop in range(num_feat * n_pst_dys + 1, num_feat * (n_pst_dys + n_nxt_dys), 1):
        if col_drop % num_feat != 0:
            col_drop_list.append(col_drop)
    #
    # for col_drop in range(num_feat * n_pst_dys + n_nxt_dys, num_feat * (n_pst_dys + n_nxt_dys), 1):
    #     if col_drop % num_feat != 0:
    #         col_drop_list.append(col_drop)

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


def save_model_summary(model):
    train_save_num = 1
    file_model_sum = f'{cwd}/resource/models/summaries/model_class_lstm_{train_save_num}.png'
    keras.utils.plot_model(model, to_file=file_model_sum,
                           show_shapes=True, show_layer_names=True,
                           show_layer_activations=True)


def save_model_compile(model):
    train_save_num = 1
    file_name_save_model = f'{cwd}/resource/models/compiled/model_class_lstm_{train_save_num}.keras'
    model.save(file_name_save_model)


def save_model_history(history, ticker='SOL-USD', start_dtt='2020-01-01', end_dtt=None, train_save_num=1):
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


def design_model(input_size, output_size):
    # model = Sequential()
    # model_input = keras.Input(shape=(n_pst_dys, num_feat))
    model_input = keras.Input(shape=input_size)
    model_layer_1 = LSTM(50, activation=keras.activations.relu, return_sequences=True)(model_input)
    model_layer_2 = Dropout(.1)(model_layer_1)
    model_layer_3 = LSTM(2, activation=keras.activations.relu)(model_layer_2)
    model_layer_4 = Dropout(.2)(model_layer_3)
    model_output = Dense(output_size, activation=keras.activations.sigmoid)(model_layer_4)
    model = keras.Model(inputs=model_input, outputs=model_output)
    save_model_summary(model)
    return model


def build_model(model, x_prep, y, test_ratio=.3, y_t=.4):
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07),
                  metrics=[keras.metrics.BinaryAccuracy(threshold=y_t),
                           keras.metrics.Recall(thresholds=y_t),
                           keras.metrics.Precision(thresholds=y_t),
                           keras.metrics.F1Score(threshold=y_t)])

    history = model.fit(x_prep, y, batch_size=20, epochs=200, validation_split=test_ratio, verbose=1)
    save_model_compile(model)
    save_model_history(history)
    return history, model


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
    # print(x_prep.shape)
    # print(y.shape)
    # return
    # input_size = (n_pst_dys, num_feat)
    input_size = (n_pst_dys, num_feat)
    # output_size = n_nxt_dys
    output_size = n_nxt_dys
    model = design_model(input_size, output_size)
    history, model = build_model(model, x_prep, y)

    # explanatory_today = df_explanatory.reshape((n_sample, look_back, num_feat))
    # forecast_tomorrow = model.predict(explanatory_today, verbose=0)

    return history.history


def back_test(data_test, money=10000, ticker='SOL-USD', start_dtt='2020-01-01', end_dtt=None, train_save_num=1):
    test_engine = Backtest(data_test, SimpleClassificationUD,
                           cash=money, commission=.002, exclusive_orders=True)
    test_history = test_engine.run()
    # test_history = test_history.to_frame(name='Values').loc[:'Return [%]']
    # test_history = test_history.to_frame(name='Values').loc[:'Return [%]']

    keep_row_list = ['Start', 'End', 'Duration', 'Return [%]', 'Volatility (Ann.) [%]', '# Trades', 'Win Rate [%]',
                     'Profit Factor']

    test_history = test_history.loc[keep_row_list]

    final_money = money * test_history.at['Return [%]']
    test_history = pd.concat(
        [pd.Series(data=[money, final_money], index=['Initial Money', 'Final Money']), test_history], axis=0)
    # last_item = {"final money": money}
    # test_history.insert()

    if end_dtt is None:
        end_dtt = str(datetime.date.today())

    start_dtt = datetime.datetime.strptime(start_dtt, '%Y-%m-%d').strftime('%y_%m_%d')
    end_dtt = datetime.datetime.strptime(end_dtt, '%Y-%m-%d').strftime('%y_%m_%d')

    ticker = ticker.replace("-", "_").lower()
    file_hist = f'{cwd}/resource/reports/histories/test/data_hist_test_ti_{ticker}_st_{start_dtt}_ed_{end_dtt}_n_{train_save_num}.csv'
    test_history.to_csv(file_hist)

    return test_history


def get_bito_bal(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)

    # Set trading pair to btc_usdt
    # pair = "btc_usdt"
    # pair = "sol_twd"
    bal = bitopro_client.get_account_balance()
    bal_data = bal['data']
    bal_data_trade = [{k: v for k, v in i_bal.items()
                       if any(k_keep in k
                              for k_keep in ['currency', 'amount', 'available'])}
                      for i_bal in bal_data if float(i_bal['amount']) != 0]
    print(bal_data_trade)
    # Return non zero amount currency
    return bal_data_trade


def set_bito_single_buy(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)
    # pair = "btc_usdt"
    pair_name = "sol_twd"
    pair_action = "buy"
    # {"error":"Invalid amount 0.0001 less than min 0.01."}
    pair_amount = '0.05'
    pair_price = '200'
    # Limit buy order
    r = bitopro_client.create_an_order(pair=pair_name,
                                       action=pair_action,
                                       amount=pair_amount,
                                       price=pair_price,
                                       type=OrderType.Limit)

    # print(r)
    ret_id = 0
    if 'orderId' in r:
        ret_id = float(r['orderId'])

    print(ret_id)
    return ret_id


def set_bito_single_sell(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)
    # pair = "btc_usdt"
    pair_name = "matic_twd"
    pair_action = "sell"
    # {"error":"Invalid amount 0.02 less than min 1."}
    pair_amount = '1'
    pair_price = '100'
    # Limit buy order
    r = bitopro_client.create_an_order(pair=pair_name,
                                       action=pair_action,
                                       amount=pair_amount,
                                       price=pair_price,
                                       type=OrderType.Limit)
    print(r)
    ret_id = 0
    if 'orderId' in r:
        ret_id = float(r['orderId'])

    print(ret_id)
    return ret_id


def set_bito_multiple_buy_sell(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)

    pair_name_buy = "sol_twd"
    pair_name_sell = "matic_twd"
    pair_action_buy = "BUY"
    pair_action_sell = "SELL"
    pair_amount_buy = 0.02
    pair_amount_sell = 1
    pair_price_buy = 200
    pair_price_sell = 100
    client_buy = 200
    client_sell = 100
    # Limit buy order
    orders_1 = {'pair': f'{pair_name_buy}',
                'action': f'{pair_action_buy}',
                'amount': str(pair_amount_buy),
                'price': str(pair_price_buy),
                'timestamp': get_current_timestamp(),
                # 'timestamp': 1717336672000,
                'clientId': client_buy,
                'type': 'LIMIT'}
    # Limit sell order
    orders_2 = {'pair': f'{pair_name_sell}',
                'action': f'{pair_action_sell}',
                'amount': str(pair_amount_sell),
                'price': str(pair_price_sell),
                'timestamp': get_current_timestamp(),
                # 'timestamp': get_current_timestamp() + 60 * 1000,
                # 'timestamp': 1717336752000,
                'clientId': client_sell,
                'type': 'LIMIT'}

    batch_orders: list = [orders_1, orders_2]
    r = bitopro_client.create_batch_order(batch_orders)
    data_res = r['data']
    all_ret = []
    print(r)
    for rr in data_res:
        ret_id = 0
        if 'orderId' in rr:
            ret_id = float(rr['orderId'])
        all_ret.append(ret_id)

    print(all_ret)
    return all_ret


def get_bito_all_orders(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)
    pair_name = "sol_twd"
    r = bitopro_client.get_all_orders(pair=pair_name, status_kind=StatusKind.OPEN, ignoreTimeLimitEnable=True)
    # r = bitopro_client.get_all_orders(status_kind=StatusKind.OPEN, ignoreTimeLimitEnable=True)
    # print(r)
    data_res = r['data']
    all_ret = []
    # print(r)
    for rr in data_res:
        ret_id = 0
        if 'id' in rr:
            ret_id = float(rr['id'])
        all_ret.append(ret_id)

    print(all_ret)
    return all_ret


def cancel_bito_all_orders(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)
    r = bitopro_client.cancel_all_orders('all')
    print(r)


# not tested
def get_orders_by_ID(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)
    order_id: str = '200'
    pair_name: str = "sol_twd"
    r = bitopro_client.get_an_order(pair=pair_name, order_id=order_id)
    print(r)

# not tested
def cancel_single_orders(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)
    pair_name: str = "sol_twd"
    r = bitopro_client.cancel_all_orders(pair_name)
    print(r)

# not tested
def cancel_multiple_orders(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)
    order_id1: str = '200'
    order_id2: str = '100000'
    pair_name: str = "sol_twd"

    order_id_lst: list = [order_id1, order_id2]
    batch_orders_dict: dict = {pair_name: order_id_lst}
    r = bitopro_client.cancel_multiple_orders(batch_orders_dict)
    print(r)

# not tested
def cancel_orders_by_ID(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)
    pair_name: str = "sol_twd"
    order_id: str = '200'
    r = bitopro_client.cancel_an_order(pair=pair_name, order_id=order_id)
    print(r)


#rewrite strategies
def set_simple_strategy(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)
    # pair_name: str = "sol_twd"
    pair_name: str = "matic_twd"
    pair_name_base: str = "sol"
    pair_name_quote: str = "twd"

    def get_trading_limit(pair):
        base_precision, quote_precision, minLimit_base_amount = None, None, None
        r = bitopro_client.get_trading_pairs()  # get pairs status
        for i in r['data']:
            if i['pair'] == pair.lower():
                base_precision = i['basePrecision']  # Decimal places for Base Currency
                quote_precision = i['quotePrecision']  # Decimal places for Quoted Currency
                minLimit_base_amount = i['minLimitBaseAmount']  # Minimum order amount
                break
        return base_precision, quote_precision, minLimit_base_amount

    def strategy(base, quote):

        pair = f"{base.lower()}_{quote.lower()}"
        # Get trading pair limitations
        base_precision, quote_precision, min_limit_base_amount = get_trading_limit(pair)
        # Query balance
        r = bitopro_client.get_account_balance()
        bal_base, bal_quote = None, None
        for curr in r['data']:
            if curr['currency'] == base.lower():
                bal_base = eval(curr['amount'])
            if curr['currency'] == quote.lower():
                bal_quote = eval(curr['amount'])
            if bal_base is not None and bal_quote is not None:
                break
        print(f"Balance: {base}: {bal_base}, {quote}: {bal_quote}")
        orderbook = bitopro_client.get_order_book(pair=pair, limit=1, scale=0)
        # Highest buying price
        bid = float(orderbook['bids'][0]['price']) if len(orderbook['bids']) > 0 else None
        # Lowest selling price
        ask = float(orderbook['asks'][0]['price']) if len(orderbook['asks']) > 0 else None
        # Mid-price
        mid = 0.5 * (bid + ask) if (bid and ask) else None
        # Place an order, limit buy order (at mid-price)
        amount = round(0.0001, base_precision)
        price = round(mid, quote_precision)
        if float(amount) >= float(min_limit_base_amount):
            r = bitopro_client.create_an_order(pair=pair, action='buy', amount=str(amount),
                                               price=str(price), type=OrderType.Limit)
            print(r)
            # Or place a market order (take order)
            # price = round(ask, quote_precision)
            # r = bitopro_client.create_an_order(pair=pair, action='buy', amount=str(amount),
            # price=str(price),type=OrderType.Limit)
            # print(r)
            order_id: str = r['orderId']
            time.sleep(2)
            # Query order
            while True:
                r = bitopro_client.get_an_order(pair=pair, order_id=order_id)
                print(r)
                if r['status'] == 2:
                    break
                else:
                    time.sleep(10)
            print('Order completed!')
            price_sell = round(float(r['avgExecutionPrice']) * (1 + 0.01), quote_precision)
            # Place a limit sell order (for profit margin)
            r = bitopro_client.create_an_order(pair=pair, action='sell', amount=str(amount),
                                               price=str(price_sell), type=OrderType.Limit)
            print(r)

    r = get_trading_limit(pair_name)
    print(r)
    # strategy(pair_name_base, pair_name_quote)

#rewrite strategies
# add a file to save trading flag record, then load it daily to check previous day stat
# this will allow not buying if already bougtht, next action should be sell first
# being able to sell when going going doing at certain margin, 15%
# sell also when profitable at certain margin
# buy if knowing coming days price will rise, if rise will be up to 15%, otherwise stay idle
# sell if knowing coming days price will fall
# need to access current buying price to be able to trigger alarm
# sell right now if dramatic price fall, only if price down compared to buying price, emergenty alert to sell
# sell if current price is making significant profit, 15%, comparing to buying price
def set_sma_strategy(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)
    # pair_name: str = "sol_twd"
    pair_name: str = "sol_twd"
    pair_name_base: str = "sol"
    pair_name_quote: str = "twd"

    def strategy(base, quote):
        pair = f"{base.lower()}_{quote.lower()}"
        signal_entry_long: bool = False
        signal_exit_long: bool = False
        resolution = CandlestickResolution._1d
        dt_string = '2023/01/01 00:00:00'
        start_ts = int(datetime.strptime(dt_string, "%Y/%m/%d %H:%M:%S").replace(
            tzinfo=datetime.timezone.utc).timestamp())
        end_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())

        sma_df = indicator("sma", pair, resolution, start_ts, end_ts, length=7)
        sma21_df = indicator("sma", pair, resolution, start_ts, end_ts, length=21)

        sma_df["SMA_21"] = sma21_df["SMA_21"]

        sma_7_last = sma_df['SMA_7'].iloc[-1]
        sma_7_penultimate = sma_df['SMA_7'].iloc[-2]

        sma_21_last = sma_df['SMA_21'].iloc[-1]
        sma_21_penultimate = sma_df['SMA_21'].iloc[-2]

        if (
                float(sma_7_penultimate) < float(sma_21_penultimate) and
                float(sma_7_last) > float(sma_21_last)
        ):
            signal_entry_long = True  # Entry signal
        elif (
                float(sma_7_penultimate) > float(sma_21_penultimate) and
                float(sma_7_last) < float(sma_21_last)
        ):
            signal_exit_long = True  # Exit signal
        if signal_entry_long:
            # Please check the balance and place a buy order
            signal_entry_long = False
        if signal_exit_long:
            # Please check the balance and place a sell order, if unable to do so, don't proceed.
            signal_exit_long = False
    # strategy(pair_name_base, pair_name_quote)



def main2():
    # ui.load_data(get_bito_bal)
    # ui.load_data(set_bito_single_buy)
    # ui.load_data(set_bito_single_sell)
    # ui.load_data(set_bito_multiple_buy_sell)
    # ui.load_data(get_bito_all_orders)
    ui.load_data(cancel_bito_all_orders)
    # print(get_current_timestamp())


def main1():
    start_date = '2020-01-01'
    # extracted_data = extract_data(start_dtt=start_date)
    # print(extracted_data)
    loaded_data = load_data(start_dtt=start_date)
    # print(loaded_data.head(6))

    n_past_days = global_past_dys
    n_nxt_days = global_next_dys
    # data_extended = process_data(extracted_data, n_pst_dys=n_past_days, n_nxt_dys=n_nxt_days)
    # print(data_extended)
    data_extended = process_data(loaded_data, n_pst_dys=n_past_days, n_nxt_dys=n_nxt_days)
    # print(data_extended.iloc[1, :])
    # print(data_extended.head(6))
    loaded_data_extended = load_data_extended(start_dtt=start_date)
    # print(loaded_data_extended.iloc[1, :])
    # train_history = train_data(data_extended, n_pst_dys=n_past_days)
    train_history = train_data(loaded_data_extended.head(50), n_pst_dys=n_past_days, n_nxt_dys=n_nxt_days)
    invest_money = 10000
    # test_history = back_test(data_extended, money=invest_money)
    test_history = back_test(loaded_data_extended.head(10),
                             money=invest_money)
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
    main2()

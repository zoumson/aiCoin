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

cwd = os.getcwd()


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
    return agg


# convert series to supervised learning
def series_to_feat(data_n):
    n_vars = 1 if type(data_n) is list else data_n.shape[1]
    n_in = 1 if type(data_n) is list else data_n.shape[0]
    df = DataFrame(data_n)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(0, n_in, 1):
        cols = np.concatenate([cols, df.iloc[i]])
        names += [f"var{j + 1}(t-{n_in - i})" for j in range(n_vars)]
    agg = pd.DataFrame(columns=names)
    agg.loc[len(agg)] = cols
    return agg


def model_nn_1(dataset,
               file_name_save_model, file_name_save_scaler_min_max, file_name_save_scaler_pca,
               mum_past_days=1, mum_next_days=1, pca_component_offset=2, train_ratio=.7,
               epoc_size=50, bat_size=20,
               num_lstm=None,
               prob_drop=None):
    # load dataset
    if prob_drop is None:
        prob_drop = [.1, .2, .3, .4]
    if num_lstm is None:
        num_lstm = [100, 80, 50, 30]
    values_to_build_past_next_relationship = dataset.values
    # ensure all data is float
    values_to_build_past_next_relationship = values_to_build_past_next_relationship.astype('float32')
    num_feat = values_to_build_past_next_relationship.shape[1]

    # frame as supervised learning
    values_with_past_next_relationship = series_to_supervised(values_to_build_past_next_relationship,
                                                              mum_past_days, mum_next_days)
    col_drop_list = []
    for col_drop in range(num_feat * mum_past_days + 1, num_feat * (mum_next_days + mum_past_days), 1):
        if col_drop % num_feat != 0:
            col_drop_list.append(col_drop)

    values_with_past_next_relationship.drop(values_with_past_next_relationship.columns[col_drop_list],
                                            axis=1,
                                            inplace=True)

    # return
    # values_with_past_next_relationship.loc[values_with_past_next_relationship['var1(t)'] >= 0, 'var1(t)'] = 1
    # values_with_past_next_relationship.loc[values_with_past_next_relationship['var1(t)'] < 0, 'var1(t)'] = 0
    # print(values_with_past_next_relationship.head(5))
    # return

    # split into train and test sets
    values_with_past_next_relationship = values_with_past_next_relationship.values

    train_size = int(len(values_with_past_next_relationship) * train_ratio)
    test_size = len(dataset) - train_size

    # increase feature and use pca to detect or retrieve important features
    pca_component_offset = 0
    pca_component = num_feat * mum_past_days - pca_component_offset
    scaler_pca = PCA(n_components=pca_component)
    # pca applies only on feature not only y values
    features_with_past_next_relationship = values_with_past_next_relationship[:, :-mum_next_days]
    main_val_to_predict = values_with_past_next_relationship[:, -mum_next_days:]
    main_val_to_predict = main_val_to_predict.reshape(len(main_val_to_predict), mum_next_days)

    scaler_min_max_x_1 = MinMaxScaler(feature_range=(0, 1))
    scaler_min_max_x_2 = MinMaxScaler(feature_range=(0, 1))

    scaler_min_max_y_1 = MinMaxScaler(feature_range=(0, 1))
    scaler_min_max_y_2 = MinMaxScaler(feature_range=(0, 1))

    features_with_past_next_relationship_norm_x_1 = scaler_min_max_x_1.fit_transform(
        features_with_past_next_relationship)
    # features_with_past_next_relationship_norm_x_1_pca = scaler_pca.fit_transform(
    #     features_with_past_next_relationship_norm_x_1)

    features_with_past_next_relationship_norm_x_1_pca = features_with_past_next_relationship_norm_x_1
    # main_val_to_predict_norm_y_1 = scaler_min_max_y_1.fit_transform(main_val_to_predict)
    main_val_to_predict_norm_y_1 = main_val_to_predict

    # features_with_past_next_relationship_norm_x_1_pca_norm_x_2 = scaler_min_max_x_2.fit_transform(
    #     features_with_past_next_relationship_norm_x_1_pca)

    features_with_past_next_relationship_norm_x_1_pca_norm_x_2 = features_with_past_next_relationship_norm_x_1_pca
    values_with_past_next_relationship_norm_1_x_pca_norm_2_x_norm_1_y = np.concatenate(
        (features_with_past_next_relationship_norm_x_1_pca_norm_x_2,
         main_val_to_predict_norm_y_1),
        axis=1)

    # scaler_min_max = MinMaxScaler(feature_range=(0, 1))
    # values_with_past_next_relationship_pca_norm = scaler_min_max.fit_transform(values_with_past_next_relationship_pca)

    train_x = features_with_past_next_relationship[:train_size, :-mum_next_days]
    test_x = features_with_past_next_relationship[train_size:, :-mum_next_days]

    train_y = main_val_to_predict[:train_size, -mum_next_days:]
    test_y = main_val_to_predict[train_size:, -mum_next_days:]

    train_x_pca_norm = values_with_past_next_relationship_norm_1_x_pca_norm_2_x_norm_1_y[:train_size, :-mum_next_days]
    test_x_pca_norm = values_with_past_next_relationship_norm_1_x_pca_norm_2_x_norm_1_y[train_size:, :-mum_next_days]

    train_y_norm = values_with_past_next_relationship_norm_1_x_pca_norm_2_x_norm_1_y[:train_size, -mum_next_days:]
    test_y_norm = values_with_past_next_relationship_norm_1_x_pca_norm_2_x_norm_1_y[train_size:, -mum_next_days:]

    # reshape input to be 3D [samples, timesteps, features]
    train_x_pca_norm = train_x_pca_norm.reshape((train_x_pca_norm.shape[0], 1,
                                                 train_x_pca_norm.shape[1]))
    test_x_pca_norm = test_x_pca_norm.reshape((test_x_pca_norm.shape[0], 1,
                                               test_x_pca_norm.shape[1]))
    #
    # design network
    model = Sequential()
    model.add(LSTM(50,activation='tanh',
                   # input_shape=(train_x_pca_norm.shape[0], train_x_pca_norm.shape[1])))
                   input_shape=(train_x_pca_norm.shape[1], train_x_pca_norm.shape[2])))
    # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(1, activation='tanh'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(train_x_pca_norm, train_y_norm, epochs=2, batch_size=2)
    # Final evaluation of the model
    scores = model.evaluate(test_x_pca_norm, test_y_norm, verbose=0)
    Y_test_pred = np.array(model.predict(test_x_pca_norm))
    # Y_test_pred = 1*np.array(tf.greater(Y_test_pred, .5))
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    pos = [sum(y >= 0 for y in x) for x in zip(*Y_test_pred)]
    neg = [len(Y_test_pred) - x for x in pos]
    print(pos, neg)
    print(Y_test_pred)
    return
    #
    # model.add(LSTM(num_lstm[0], activation='tanh',
    #                input_shape=(train_x_pca_norm.shape[1], train_x_pca_norm.shape[2]),
    #                return_sequences=True))
    # model.add(LSTM(200, activation='tanh', input_shape=(train_x_pca_norm.shape[1], train_x_pca_norm.shape[2])))
    # model.add(LSTM(200, input_shape=(train_x_pca_norm.shape[1], train_x_pca_norm.shape[2]),
    #                return_sequences=True))
    # model.add(Dropout(.1))

    # model.add(Dropout(.3))
    #     model.add(LSTM(num_lstm[i], activation='tanh', return_sequences=True))
    #     model.add(Dropout(prob_drop[i]))

    # model.add(Dropout(prob_drop[0]))
    #
    # for i in range(1, len(num_lstm)-1):
    #     model.add(LSTM(num_lstm[i], activation='tanh', return_sequences=True))
    #     model.add(Dropout(prob_drop[i]))
    #
    # model.add(LSTM(num_lstm[-1], activation='tanh'))
    # model.add(Dropout(prob_drop[-1]))

    # model.add(Dense(mum_next_days))
    # model.add(Dense(1))
    # model.compile(loss='mae', optimizer='adam')
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    history = model.fit(train_x_pca_norm, train_y_norm,
                        epochs=epoc_size, batch_size=bat_size,
                        validation_data=(test_x_pca_norm, test_y_norm),
                        verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    test_y_pred_norm = model.predict(test_x_pca_norm)

    # test_y_pred = scaler_min_max_y_1.inverse_transform(test_y_pred_norm)
    test_y_pred = test_y_pred_norm
    cc = np.sum(np.array(test_y_pred) >= 0, axis=0)
    dd = np.sum(np.array(test_y) >= 0, axis=0)
    test_y_y = np.concatenate((test_y_pred, test_y), axis=1)
    print(test_y_y)
    print(cc)
    print(dd)
    # # calculate RMSE
    rmse = sqrt(mean_squared_error(test_y, test_y_pred))

    model.save(file_name_save_model)

    file_name_save_scaler_min_max_x_1 = f"{file_name_save_scaler_min_max}_x_1.pkl"
    file_name_save_scaler_min_max_x_2 = f"{file_name_save_scaler_min_max}_x_2.pkl"
    file_name_save_scaler_min_max_y_1 = f"{file_name_save_scaler_min_max}_y_1.pkl"

    file_name_save_scaler_all = [file_name_save_scaler_min_max_x_1,
                                 file_name_save_scaler_min_max_x_2,
                                 file_name_save_scaler_min_max_y_1,
                                 file_name_save_scaler_pca]

    scaler_all = [scaler_min_max_x_1,
                  scaler_min_max_x_2,
                  scaler_min_max_y_1,
                  scaler_pca]

    for file_name, scal in zip(file_name_save_scaler_all, scaler_all):
        with open(file_name, 'wb') as f:
            pickle.dump(scal, f)

    print('Test RMSE: %.3f' % rmse)


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
            r = bitopro_client.create_an_order(pair=pair, action='buy', amount=str(amount), price=str(price),
                                               type=OrderType.Limit)
            print(r)
            # Or place a market order (take order) price = round(ask, quote_precision) r =
            # bitopro_client.create_an_order(pair=pair, action='buy', amount=str(amount), price=str(price),
            # type=OrderType.Limit) print(r)
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
            r = bitopro_client.create_an_order(pair=pair, action='sell', amount=str(amount), price=str(price_sell),
                                               type=OrderType.Limit)
            print(r)

    r = get_trading_limit(pair_name)
    print(r)
    # strategy(pair_name_base, pair_name_quote)


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
        start_ts = int(
            datetime.strptime(dt_string, "%Y/%m/%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp())
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


def get_hist_yahoo():
    ticker = 'SOL-USD'
    # start_dt = '2024-01-01'
    start_dt = '2020-01-01'
    end_dt = '2024-04-01'

    # data_frame = yf.download(tickers=ticker, start='2022-03-10', end='2022-03-11', prepost=True, progress=False)
    data_frame = yf.download(tickers=ticker, start=start_dt, prepost=True, progress=False)
    # data_frame = yf.download(ticker)
    data_frame = data_frame.drop(columns='Adj Close')
    # print("Yahoo")
    # print(len(data_frame))
    # print(data_frame)
    return data_frame


def get_hist_bito(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)

    # Set trading pair to btc_usdt
    # pair = "btc_usdt"
    pair = "sol_twd"
    # '2024-01-01'
    # dt_string = '2024/01/01 00:00:00'
    dt_string = '2020/01/01 00:00:00'
    start_ts = int(
        datetime.datetime.strptime(dt_string, "%Y/%m/%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp())

    end_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    response = bitopro_client.get_candlestick(pair, CandlestickResolution._1d, start_ts, end_ts)
    print(response)
    print(len(response['data']))  # Retrieve the number of candlesticks

    df = pd.DataFrame(response['data'])
    # print("Bito")
    # print(len(df))
    # print(df.head())

    # df_yahoo = get_hist_yahoo()
    df["Datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("Datetime", inplace=True)
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    return df
    # print(df[['Close']])
    # data2 = pd.DataFrame()
    # data2['Close1'] = df[['Close']]
    # data2['Close2'] = df_yahoo[['Close']]
    # data1 = df_yahoo[['Close']]
    # print(np.shape(df[['Close']]))
    # print(np.shape(data2))
    # dff = pd.DataFrame(data1, data2)
    # print(df.dtypes)
    # print(df.tail())
    #
    # # Plotting the closing price trend
    # df[['Close']].plot(figsize=(15, 8))
    # df[['Close']].plot(figsize=(15, 8))
    # df_yahoo[['Close']].plot(figsize=(15, 8))

    # dff = pd.DataFrame(df[['Close']], df_yahoo[['Close']])
    # scalers = MinMaxScaler()
    # scalers = scalers.fit(data2)
    # data2 = scalers.transform(data2)
    # data2 = pd.DataFrame(data2)
    # data2.plot()
    # plt.show()


def get_hist(_email, _api_secret, _api_key):
    df_bito = get_hist_bito(_email, _api_secret, _api_key)
    df_yahoo = get_hist_yahoo()

    data = pd.DataFrame()
    data['Close1'] = df_bito[['Close']]
    data['Close2'] = df_yahoo[['Close']]
    scaler = MinMaxScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)
    data = pd.DataFrame(data)
    data.plot()
    plt.show()


def preprocess_1(df):
    """
    Create the target variable
    """
    # print(df.head(3))
    df['Close'] = df['Close'].pct_change(-1)
    df.Close = df.Close * -1
    # df.Close = df.Close * 100
    # df.Close = df.Close *

    # Remove rows with any missing data
    df = df.dropna().copy()
    # a = df.shape[0]
    # df.drop([a - 1, a - 2, a - 3])
    # df.drop([0, 1])
    print(df.head(10))
    # print(df.index)
    # print(df.loc[['2020-04-10']])
    return df
def preprocess_2(df):
    """
    Create the target variable
    """
    # print(df.head(3))
    df['Close'] = df['Close'].pct_change(-1)
    df.Close = df.Close * -1
    df.Close = df.Close * 100

    # Remove rows with any missing data
    df = df.dropna().copy()
    # a = df.shape[0]
    # df.drop([a - 1, a - 2, a - 3])
    # df.drop([0, 1])
    df.loc[df.Close >= 0, 'Close'] = 1
    df.loc[df.Close < 0, 'Close'] = -1
    print(df.head(10))

    # print(df.index)
    # print(df.loc[['2020-04-10']])
    return df


def get_last_n(data_in, n=1):
    # return data_in.iloc[[5]]
    # return data_in.loc[['2020-04-10']]
    return data_in.dropna().tail(n)


def get_date_data(data_in, data_str):
    # return data_in.iloc[[5]]
    return data_in.loc[[data_str]]
    # return data_in.dropna().tail(n)


def get_feat_last_n(data_n, file_name_save_scaler_pca, file_name_save_scaler_min_max):
    file_name_save_scaler_min_max_x_1 = f"{file_name_save_scaler_min_max}_x_1.pkl"
    file_name_save_scaler_min_max_x_2 = f"{file_name_save_scaler_min_max}_x_2.pkl"
    file_name_save_scaler_min_max_y_1 = f"{file_name_save_scaler_min_max}_y_1.pkl"

    file_name_save_scaler_all = [file_name_save_scaler_min_max_x_1,
                                 file_name_save_scaler_min_max_x_2,
                                 file_name_save_scaler_min_max_y_1,
                                 file_name_save_scaler_pca]
    scaler_all = []

    for file_name in file_name_save_scaler_all:
        with open(file_name, 'rb') as f:
            scal = pickle.load(f)
            scaler_all.append(scal)

    scaler_all = np.array(scaler_all)

    data_n_val = data_n.values

    data_n_val = data_n_val.astype('float32')

    data_n_val_feat = series_to_feat(data_n_val)

    data_n_val_feat = data_n_val_feat.values
    scal_min_max_x_1 = scaler_all[0]
    data_n_val_feat_norm_1 = scal_min_max_x_1.transform(data_n_val_feat)

    scal_pca = scaler_all[-1]
    data_n_val_feat_norm_1_pca = scal_pca.transform(data_n_val_feat_norm_1)

    scal_min_max_x_2 = scaler_all[1]
    data_n_val_feat_norm_1_pca_norm2 = scal_min_max_x_2.transform(data_n_val_feat_norm_1_pca)
    return data_n_val_feat_norm_1_pca_norm2


def predict_1(data_feat_in, file_name_model, file_name_scaler_min_max):
    file_name_save_scaler_min_max_x_1 = f"{file_name_scaler_min_max}_x_1.pkl"
    file_name_save_scaler_min_max_x_2 = f"{file_name_scaler_min_max}_x_2.pkl"
    file_name_save_scaler_min_max_y_1 = f"{file_name_scaler_min_max}_y_1.pkl"

    file_name_save_scaler_all = [file_name_save_scaler_min_max_x_1,
                                 file_name_save_scaler_min_max_x_2,
                                 file_name_save_scaler_min_max_y_1]
    scaler_all = []

    for file_name in file_name_save_scaler_all:
        with open(file_name, 'rb') as f:
            scal = pickle.load(f)
            scaler_all.append(scal)

    scaler_all = np.array(scaler_all)
    scaler_min_max = scaler_all[-1]
    model = load_model(file_name_model)
    # with open(file_name_scaler_min_max, 'rb') as f:
    #     scaler_min_max = pickle.load(f)

    # make a prediction
    data_feat_in_pred = data_feat_in.reshape((data_feat_in.shape[0], 1,
                                              data_feat_in.shape[1]))
    test_y_pred_norm = model.predict(data_feat_in_pred)

    test_y_pred = scaler_min_max.inverse_transform(test_y_pred_norm)

    print(test_y_pred)
    return test_y_pred

def ex_lstm_class():
    # LSTM for sequence classification in the IMDB dataset
    import tensorflow as tf
    from keras.datasets import imdb
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Embedding
    from keras.preprocessing import sequence
    # fix random seed for reproducibility
    tf.random.set_seed(7)
    # load the dataset but only keep the top n words, zero the rest
    # top_words = 5000
    top_words = 2
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    print(len(X_train[5]))
    print(y_train[6])
    # print(y_train)
    return
    # truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    # model.add(Embedding(top_words, embedding_vecor_length, input_dim=max_review_length))
    model.add(Embedding(2, embedding_vecor_length))
    model.add(LSTM(2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=2, batch_size=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    Y_test_pred = model.predict(X_test)

    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print(Y_test_pred)


def main():
    # ex_lstm_class()
    # return
    train_save_num = 1
    file_save_mod = f'{cwd}/resource/models/model_lstm_{train_save_num}.keras'
    file_save_scal_min_max = f'{cwd}/resource/scalers/scaler_min_max_{train_save_num}'
    file_save_scal_pca = f'{cwd}/resource/scalers/scaler_pca_{train_save_num}.pkl'

    train_load_num = 2
    file_load_mod = f'{cwd}/resource/models/model_lstm_{train_load_num}.keras'
    file_load_scal_min_max = f'{cwd}/resource/scalers/scaler_min_max_{train_load_num}.pkl'
    # file_load_scal_pca = f'{cwd}/resource/scalers/scaler_pca_{train_load_num}.pkl'
    file_load_scal_pca = file_save_scal_pca

    # ui.load_data(set_single_buy)
    # ui.load_data(get_hist)
    data_here = get_hist_yahoo()
    # modelNN1(data_here[['Close']])
    # modelNN2(data_here[['Close']])
    # print(data_here['2020-04-12'])
    data_here.insert(0, 'Close', data_here.pop('Close'))
    data_here = preprocess_1(data_here)
    # data_here = preprocess_2(data_here)
    # print(data_here['2020-04-12'])

    model_nn_1(data_here, file_save_mod, file_save_scal_min_max, file_save_scal_pca)
    # last_n = 1
    # # past_n_data = get_last_n(data_here, last_n)
    # past_date_data = get_date_data(data_here, '2020-04-12')
    # print(past_date_data)
    # feat_last_n = get_feat_last_n(past_date_data, file_load_scal_pca, file_save_scal_min_max)
    # # # # # print(past_n_data_feat)
    # pred_data = predict_1(feat_last_n, file_save_mod, file_save_scal_min_max)
    # # # print(pred_data)


if __name__ == "__main__":
    main()

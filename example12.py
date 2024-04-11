import datetime
import time

from bitoproClient.bitopro_indicator import indicator
from bitoproClient.bitopro_restful_client import BitoproRestfulClient, CandlestickResolution, OrderType, StatusKind
from bitoproClient.bitopro_util import get_current_timestamp
from sklearn.preprocessing import MinMaxScaler

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


def modelNN1(dataset):
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
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.decomposition import PCA

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

    # load dataset
    # dataset = read_csv('pollution.csv', header=0, index_col=0)
    # print(dataset.head)
    # dataset.insert(0, 'Close', dataset.pop('Close'))
    values = dataset.values
    # integer encode direction
    # encoder = LabelEncoder()
    # values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # print(scaled)
    # frame as supervised learning
    # use 7 past day data to predict current pricing
    reframed = series_to_supervised(scaled, 7, 1)
    # drop columns we don't want to predict
    # reframed.drop(reframed.columns[[6, 7, 8, 9]], axis=1, inplace=True)
    # drop current unused data, first column is the actual price in the original data
    # tranform data has last column as price data to predict
    reframed.drop(reframed.columns[[36, 37, 38, 39]], axis=1, inplace=True)
    # remove_prob = .1
    # remove_feat = VarianceThreshold(threshold=(remove_prob * (1 - remove_prob)))
    # reframed2 = remove_feat.fit_transform(reframed)
    # print(reframed.head())
    # print(reframed2.head())
    #
    # split into train and test sets
    values = reframed.values

    train_size = int(len(reframed) * 0.7)
    test_size = len(dataset) - train_size
    # attempt to reduce feature via variance thresholding
    # remove_prob = .8
    # remove_feat = VarianceThreshold(threshold=(remove_prob * (1 - remove_prob)))
    # remove_feat = VarianceThreshold()

    # increase feature and use pca to detect or retrieve important features
    pca = PCA(n_components=15)
    # pca applies only on feature not only y values
    values_2 = values[:, :-1]
    # reframed2 = remove_feat.fit_transform(values_2)
    values_2 = pca.fit_transform(values_2)
    # values_2 = pca.singular_values_

    values_2 = np.concatenate((values_2[:, :],
                               values[:, -1].reshape((values[:, -1].shape[0], 1))),
                              axis=1)
    # values_2 = concatenate((values_2, values[:, -1]), axis=1)
    # normalize features
    # scaler2 = MinMaxScaler(feature_range=(0, 1))
    # scaler2.fit(values_2)
    # print(remove_feat.variances_)
    # train = values[:train_size, :]
    # test = values[train_size:, :]

    train = values_2[:train_size, :]
    test = values_2[train_size:, :]
    # split into input and outputs
    train_x, train_y = train[:, :-1], train[:, -1]
    # train_x, train_y = train[:, :-1], train[:, -1]
    # train_x, train_y = values_2[:train_size, :], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    # test_x, test_y = values_2[train_size:, :], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    # print(train_x.shape, train_y.shape, test_X.shape, test_y.shape)
    #
    # design network
    model = Sequential()
    model.add(LSTM(5, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_x, train_y,
                        epochs=5, batch_size=2,
                        validation_data=(test_x, test_y),
                        verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    yhat = model.predict(test_x)

    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
    # invert scaling for forecast
    # inv_yhat = concatenate((yhat, test_x[:, 1:]), axis=1)

    inv_yhat = np.concatenate((pca.inverse_transform(test_x), yhat.reshape((len(yhat), 1))), axis=1)
    # inv_yhat = concatenate((yhat, pca.inverse_transform(test_x)), axis=1)

    # normalize features
    # inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    # inv_yhat = scaler2.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    # inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
    # inv_y = concatenate((test_y, pca.inverse_transform(test_x)), axis=1)
    inv_y = np.concatenate((pca.inverse_transform(test_x), test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    # inv_y = scaler2.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
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
    # scaler = MinMaxScaler()
    # scaler = scaler.fit(data2)
    # data2 = scaler.transform(data2)
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


def main():
    # ui.load_data(set_single_buy)
    # ui.load_data(get_hist)
    data_here = get_hist_yahoo()
    # modelNN1(data_here[['Close']])
    # modelNN2(data_here[['Close']])
    data_here.insert(0, 'Close', data_here.pop('Close'))
    modelNN1(data_here)


if __name__ == "__main__":
    main()

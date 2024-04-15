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

# scaler_filename = "scaler.save"
# joblib.dump(scaler, scaler_filename)

# And now to load...

# scaler = joblib.load(scaler_filename)
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


def modelNN1(dataset):
    # from math import sqrt
    # from numpy import concatenate
    # from matplotlib import pyplot
    # # from pandas import read_csv
    # from pandas import DataFrame
    # from pandas import concat
    # from sklearn.preprocessing import MinMaxScaler
    # from sklearn.preprocessing import LabelEncoder
    # from sklearn.metrics import mean_squared_error
    # from keras.models import Sequential
    # # from keras.layers import Dense
    # from keras.layers import LSTM, Dropout, Dense
    # from sklearn.feature_selection import VarianceThreshold
    # from sklearn.decomposition import PCA

    # load dataset
    # dataset = read_csv('pollution.csv', header=0, index_col=0)
    values_to_build_past_next_relationship = dataset.values
    # ensure all data is float
    values_to_build_past_next_relationship = values_to_build_past_next_relationship.astype('float32')
    num_feat = values_to_build_past_next_relationship.shape[1]

    # normalize features
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled = scaler.fit_transform(values)
    # print(scaled)
    # frame as supervised learning
    # use 7 past day data to predict current pricing
    # reframed = series_to_supervised(scaled, 7, 1)
    # 7 working days prediction, monday to friday
    mum_past_days = 7
    mum_next_days = 7
    values_with_past_next_relationship = series_to_supervised(values_to_build_past_next_relationship,
                                                              mum_past_days, mum_next_days)
    # drop columns we don't want to predict
    # reframed.drop(reframed.columns[[6, 7, 8, 9]], axis=1, inplace=True)
    # drop current unused data, first column is the actual price in the original data
    # tranform data has last column as price data to predict
    # values_with_past_next_relationship.drop(values_with_past_next_relationship.columns[[36, 37, 38, 39]],
    # values_with_past_next_relationship.drop(values_with_past_next_relationship.columns[[6, 7, 8, 9]],
    # values_with_past_next_relationship.drop(values_with_past_next_relationship.columns[[6, 7, 8, 9]],
    #                                         axis=1,
    #                                         inplace=True)
    # num_feat * (mum_next_days + mum_past_days) - 1
    col_drop_list = []
    for col_drop in range(num_feat * mum_past_days + 1, num_feat * (mum_next_days + mum_past_days), 1):
        if col_drop % num_feat != 0:
            col_drop_list.append(col_drop)

    values_with_past_next_relationship.drop(values_with_past_next_relationship.columns[col_drop_list],
                                            axis=1,
                                            inplace=True)

    # print(values_with_past_next_relationship.iloc[0:3, -11:-5])
    # return
    # split into train and test sets
    values_with_past_next_relationship = values_with_past_next_relationship.values

    train_size = int(len(values_with_past_next_relationship) * 0.7)
    test_size = len(dataset) - train_size
    # attempt to reduce feature via variance thresholding
    # remove_prob = .8
    # remove_feat = VarianceThreshold(threshold=(remove_prob * (1 - remove_prob)))
    # remove_feat = VarianceThreshold()

    # increase feature and use pca to detect or retrieve important features
    pca_component = num_feat * mum_past_days - 2
    scaler_pca = PCA(n_components=pca_component)
    # pca applies only on feature not only y values
    features_with_past_next_relationship = values_with_past_next_relationship[:, :-mum_next_days]
    main_val_to_predict = values_with_past_next_relationship[:, -mum_next_days:]
    main_val_to_predict = main_val_to_predict.reshape(len(main_val_to_predict), mum_next_days)

    features_with_past_next_relationship_pca = scaler_pca.fit_transform(features_with_past_next_relationship)

    values_with_past_next_relationship_pca = np.concatenate((features_with_past_next_relationship,
                                                             main_val_to_predict),
                                                            axis=1)

    scaler_min_max = MinMaxScaler(feature_range=(0, 1))
    values_with_past_next_relationship_pca_norm = scaler_min_max.fit_transform(values_with_past_next_relationship_pca)

    train_x = features_with_past_next_relationship[:train_size, :-mum_next_days]
    test_x = features_with_past_next_relationship[train_size:, :-mum_next_days]

    train_y = main_val_to_predict[:train_size, -mum_next_days:]
    test_y = main_val_to_predict[train_size:, -mum_next_days:]

    train_x_pca_norm = values_with_past_next_relationship_pca_norm[:train_size, :-mum_next_days]
    test_x_pca_norm = values_with_past_next_relationship_pca_norm[train_size:, :-mum_next_days]

    train_y_norm = values_with_past_next_relationship_pca_norm[:train_size, -mum_next_days:]
    test_y_norm = values_with_past_next_relationship_pca_norm[train_size:, -mum_next_days:]

    # reshape input to be 3D [samples, timesteps, features]
    train_x_pca_norm = train_x_pca_norm.reshape((train_x_pca_norm.shape[0], 1,
                                                 train_x_pca_norm.shape[1]))
    test_x_pca_norm = test_x_pca_norm.reshape((test_x_pca_norm.shape[0], 1,
                                               test_x_pca_norm.shape[1]))
    #
    # design network
    model = Sequential()

    num_lstm_1_unit = 100
    num_lstm_2_unit = 80
    num_lstm_3_unit = 50
    num_lstm_4_unit = 30

    prob_drop_1_unit = .1
    prob_drop_2_unit = .2
    prob_drop_3_unit = .3
    prob_drop_4_unit = .4

    model.add(LSTM(num_lstm_1_unit,
                   input_shape=(train_x_pca_norm.shape[1], train_x_pca_norm.shape[2]),
                   return_sequences=True))

    model.add(Dropout(prob_drop_1_unit))

    model.add(LSTM(num_lstm_2_unit, return_sequences=True))
    model.add(Dropout(prob_drop_2_unit))

    model.add(LSTM(num_lstm_3_unit, return_sequences=True))
    model.add(Dropout(prob_drop_3_unit))

    model.add(LSTM(num_lstm_4_unit))
    model.add(Dropout(prob_drop_4_unit))

    model.add(Dense(mum_next_days))
    model.compile(loss='mse', optimizer='adam')

    # fit network
    history = model.fit(train_x_pca_norm, train_y_norm,
                        epochs=50, batch_size=20,
                        validation_data=(test_x_pca_norm, test_y_norm),
                        verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    test_y_pred_norm = model.predict(test_x_pca_norm)

    test_x_pca_norm = test_x_pca_norm.reshape((test_x_pca_norm.shape[0],
                                               test_x_pca_norm.shape[2]))

    test_y_pred_norm = test_y_pred_norm.reshape((len(test_y_pred_norm), mum_next_days))

    test_values_pred_pca_norm = np.concatenate((test_x_pca_norm, test_y_pred_norm), axis=1)

    # # # invert scaling for actual
    test_values_pred_pca = scaler_min_max.inverse_transform(test_values_pred_pca_norm)

    test_y_pred = test_values_pred_pca[:, -mum_next_days:]

    # # calculate RMSE
    rmse = sqrt(mean_squared_error(test_y, test_y_pred))

    model.save(f'{cwd}/resource/models/model_lstm_1_regression.keras')
    with open(f'{cwd}/resource/models/scaler_min_max_1.pkl', 'wb') as f:
        pickle.dump(scaler_min_max, f)
    with open(f'{cwd}/resource/models/scaler_pca_1.pkl', 'wb') as f:
        pickle.dump(scaler_pca, f)
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


def preprocess_1(df):
    """
    Create the target variable
    """
    df['Close'] = df['Close'].pct_change(-1)
    df.Close = df.Close * -1
    df.Close = df.Close * 100
    # Remove rows with any missing data
    df = df.dropna().copy()
    # print(df.head)


def get_last_7(data_in):
    data_raw_7 = data_in.iloc[-15:, :]
    print(data_raw_7)
    return data_raw_7


def get_feat_last_7(data_7):
    # data_raw_7 = data_here.iloc[-8:-1, :]
    # print(data_raw_7)
    values_to_build_past_next_relationship = data_7.values
    # ensure all data is float
    values_to_build_past_next_relationship = values_to_build_past_next_relationship.astype('float32')
    num_feat = values_to_build_past_next_relationship.shape[1]

    # 7 working days prediction, monday to friday
    mum_past_days = 7
    mum_next_days = 1
    values_with_past_next_relationship = series_to_supervised(values_to_build_past_next_relationship,
                                                              mum_past_days, mum_next_days)
    values_with_past_next_relationship = values_with_past_next_relationship.tail(1)

    col_drop_list = []
    for col_drop in range(num_feat * mum_past_days, num_feat * (mum_next_days + mum_past_days), 1):
        col_drop_list.append(col_drop)

    values_with_past_next_relationship.drop(values_with_past_next_relationship.columns[col_drop_list],
                                            axis=1,
                                            inplace=True)
    # increase feature and use pca to detect or retrieve important features
    with open(f'{cwd}/resource/models/scaler_pca_1.pkl', 'rb') as f:
        scaler_pca = pickle.load(f)
    # pca applies only on feature not only y values
    features_with_past_next_relationship = values_with_past_next_relationship.values
    print(features_with_past_next_relationship.shape)

    features_with_past_next_relationship_pca = scaler_pca.transform(features_with_past_next_relationship)

    scaler_min_max_loc = MinMaxScaler(feature_range=(0, 1))
    values_with_past_next_relationship_pca_norm = scaler_min_max_loc.fit_transform(features_with_past_next_relationship)

    return values_with_past_next_relationship_pca_norm


def predict_1(data_feat_in):
    # model.save(f'{cwd}/resource/models/model_lstm_1_regression..keras')
    model = load_model(f'{cwd}/resource/models/model_lstm_1_regression.keras')
    with open(f'{cwd}/resource/models/scaler_min_max_1.pkl', 'rb') as f:
        scaler_min_max = pickle.load(f)
    mum_next_days = 7
    data_feat_in
    # make a prediction
    data_feat_in_pred = data_feat_in.reshape((data_feat_in.shape[0], 1,
                                              data_feat_in.shape[1]))
    test_y_pred_norm = model.predict(data_feat_in_pred)

    test_x_pca_norm = data_feat_in

    test_y_pred_norm = test_y_pred_norm.reshape((len(test_y_pred_norm), mum_next_days))

    test_values_pred_pca_norm = np.concatenate((test_x_pca_norm, test_y_pred_norm), axis=1)

    # # # invert scaling for actual
    test_values_pred_pca = scaler_min_max.inverse_transform(test_values_pred_pca_norm)

    test_y_pred = test_values_pred_pca[:, -mum_next_days:]

    print(test_y_pred)
    return test_y_pred


def main():
    # ui.load_data(set_single_buy)
    # ui.load_data(get_hist)
    data_here = get_hist_yahoo()
    # modelNN1(data_here[['Close']])
    # modelNN2(data_here[['Close']])
    data_here.insert(0, 'Close', data_here.pop('Close'))
    preprocess_1(data_here)

    # modelNN1(data_here)
    past_7_data = get_last_7(data_here)
    past_7_data_feat = get_feat_last_7(past_7_data)
    pred_data = predict_1(past_7_data_feat)


if __name__ == "__main__":
    main()

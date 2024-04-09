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


def modelNN1(dataframe):
    # Stacked LSTM for international airline passengers problem with memory
    # convert an array of values into a dataset matrix
    def create_dataset(data_in, past_data=1):
        data_x, data_y = [], []
        # data_x = (len(data_in) - past_data - 1)*past_data
        # data_y = (len(data_in) - past_data - 1)*1
        for idx in range(len(data_in) - past_data - 1):
            # Use past data of length past data to predict following
            # Time slot data with single length
            data_x.append(data_in[idx:(idx + past_data), 0])
            data_y.append(data_in[idx + past_data, 0])
        return np.array(data_x), np.array(data_y)

    # fix random seed for reproducibility
    tf.random.set_seed(7)
    # load the dataset
    # dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
    # dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # print(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    look_back = 1
    train_data_x, train_data_y = create_dataset(train, look_back)
    test_data_x, test_data_y = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    train_data_x = np.reshape(train_data_x, (train_data_x.shape[0], train_data_x.shape[1], 1))
    test_data_x = np.reshape(test_data_x, (test_data_x.shape[0], test_data_x.shape[1], 1))

    # create and fit the LSTM network

    batch_size = 1
    # model = Sequential()

    print(train_data_x.shape)
    # model_ = Model()
    # inputs = tf.keras.input(shape=(32, 32, 1))
    inputs = keras.Input(shape=(batch_size, train_data_x.shape[0], ))  # input layer
    x = LSTM(5)(inputs)  # hidden layer
    outputs = Dense(1)(x)  # output layer
    #
    model = Model(inputs, outputs)
    # model = Sequential(inputs, outputs)
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # model.add(LSTM(25, input_shape=(1, look_back)))
    # model.add(LSTM(25, input_shape=(1, look_back)))
    # outputs = model(inputs)
    # model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    # model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    # model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_data_x, train_data_y, epochs=10, batch_size=2, verbose=1)
    # # for i in range(100):
    # for i in range(5):
    #     model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    #     model.reset_states()
    # # make predictions
    # train_predict = model.predict(trainX, batch_size=batch_size)
    # model.reset_states()
    # test_predict = model.predict(testX, batch_size=batch_size)
    # # invert predictions
    # train_predict = scaler.inverse_transform(train_predict)
    # trainY = scaler.inverse_transform([trainY])
    # test_predict = scaler.inverse_transform(test_predict)
    # testY = scaler.inverse_transform([testY])
    # # calculate root mean squared error
    # trainScore = np.sqrt(mean_squared_error(trainY[0], train_predict[:, 0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = np.sqrt(mean_squared_error(testY[0], test_predict[:, 0]))
    # print('Test Score: %.2f RMSE' % (testScore))
    # # shift train predictions for plotting
    # trainPredictPlot = np.empty_like(dataset)
    # trainPredictPlot[:, :] = np.nan
    # trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    # # shift test predictions for plotting
    # testPredictPlot = np.empty_like(dataset)
    # testPredictPlot[:, :] = np.nan
    # testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(dataset) - 1, :] = test_predict
    # # plot baseline and predictions
    # plt.plot(scaler.inverse_transform(dataset))
    # plt.plot(trainPredictPlot)
    # plt.plot(testPredictPlot)
    # plt.show()
def modelNN2(df):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from keras.layers import Dense, Activation, Dropout
    import time  # helper libraries

    # file is downloaded from finance.yahoo.com, 1.1.1997-1.1.2017
    # training data = 1.1.1997 - 1.1.2007
    # test data = 1.1.2007 - 1.1.2017
    input_file = "DIS.csv"

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # fix random seed for reproducibility
    np.random.seed(5)

    # load the dataset
    # df = read_csv(input_file, header=None, index_col=None, delimiter=',')

    # take close price column[5]
    # all_y = df[5].values
    # dataset = all_y.reshape(-1, 1)
    dataset = df

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets, 50% test data, 50% training data
    train_size = int(len(dataset) * 0.5)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1, timestep 240
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
    model = Sequential()
    model.add(LSTM(5, input_shape=(1, look_back)))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=2, verbose=1)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.show()
    # print('testPrices:')
    testPrices = scaler.inverse_transform(dataset[test_size + look_back:])

    # print('testPredictions:')
    # print(testPredict)

    # export prediction and actual prices
    # df = pd.DataFrame(data={"prediction": np.around(list(testPredict.reshape(-1)), decimals=2),
    #                         "test_price": np.around(list(testPrices.reshape(-1)), decimals=2)})
    # df.to_csv("lstm_result.csv", sep=';', index=None)

    # plot the actual price, prediction in test data=red line, actual price=blue line
    # plt.plot(testPredictPlot)
    # plt.show()


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
    modelNN2(data_here[['Close']])


if __name__ == "__main__":
    main()

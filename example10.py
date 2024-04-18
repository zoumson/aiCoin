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
    print("Yahoo")
    print(len(data_frame))
    print(data_frame)
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


def main():
    # ui.load_data(set_single_buy)
    ui.load_data(get_hist)


if __name__ == "__main__":
    main()

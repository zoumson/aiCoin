import datetime

from bitoproClient.bitopro_restful_client import BitoproRestfulClient, CandlestickResolution
from bitoproClient.bitopro_util import get_current_timestamp
import uicredential as ui
# Here we use pandas and matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def get_hist(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)

    # Set trading pair to btc_usdt
    # pair = "btc_usdt"
    pair = "sol_twd"
    dt_string = '2024/04/01 00:00:00'
    start_ts = int(datetime.datetime.strptime(dt_string, "%Y/%m/%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp())

    end_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    response = bitopro_client.get_candlestick(pair, CandlestickResolution._5m, start_ts, end_ts)
    print(response)
    print(len(response['data']))  # Retrieve the number of candlesticks

    df = pd.DataFrame(response['data'])
    df["Datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("Datetime", inplace=True)
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    print(df.dtypes)
    print(df.tail())

    # Plotting the closing price trend
    df[['Close']].plot(figsize=(15, 8))
    plt.show()


def main():
    ui.load_data(get_hist)


if __name__ == "__main__":
    main()

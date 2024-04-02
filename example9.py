import datetime

from bitoproClient.bitopro_restful_client import BitoproRestfulClient, CandlestickResolution
from bitoproClient.bitopro_util import get_current_timestamp
import uicredential as ui
# Here we use pandas and matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def get_balance(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)

    # Set trading pair to btc_usdt
    # pair = "btc_usdt"
    # pair = "sol_twd"
    r = bitopro_client.get_account_balance()
    print(r)


def main():
    ui.load_data(get_balance)


if __name__ == "__main__":
    main()

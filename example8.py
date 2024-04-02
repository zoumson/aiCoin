import datetime

from bitoproClient.bitopro_restful_client import BitoproRestfulClient, CandlestickResolution, OrderType
from bitoproClient.bitopro_util import get_current_timestamp
import uicredential as ui
# Here we use pandas and matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def set_single_buy(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)
    # pair = "btc_usdt"
    pair_name = "sol_twd"
    pair_action = "buy"
    pair_amount = '0.0001'
    pair_price = '200'
    # Limit buy order
    r = bitopro_client.create_an_order(pair=pair_name,
                                       action=pair_action,
                                       amount=pair_amount,
                                       price=pair_price,
                                       type=OrderType.Limit)
    print(r)


def set_single_sell(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)
    # pair = "btc_usdt"
    pair_name = "sol_twd"
    pair_action = "sell"
    pair_amount = '0.0001'
    pair_price = '200'
    # Limit buy order
    r = bitopro_client.create_an_order(pair=pair_name,
                                       action=pair_action,
                                       amount=pair_amount,
                                       price=pair_price,
                                       type=OrderType.Limit)
    print(r)


def set_multiple_buy_sell(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)

    pair_name = "sol_twd"
    pair_action_buy = "BUY"
    pair_action_sell = "BUY"
    pair_amount_buy = 0.0001
    pair_amount_sell = 0.0001
    pair_price_buy = 200
    pair_price_sell = 100000
    # Limit buy order
    orders_1 = {'pair': f'{pair_name}',
                'action': f'{pair_action_buy}',
                'amount': str(pair_amount_buy),
                'price': str(pair_price_buy),
                'timestamp': get_current_timestamp(),
                'type': 'LIMIT'}
    # Limit sell order
    orders_2 = {'pair': f'{pair_name}',
                'action': f'{pair_action_sell}',
                'amount': str(pair_amount_sell),
                'price': str(pair_price_sell),
                'timestamp': get_current_timestamp() + 60,
                'type': 'LIMIT'}
    #
    # batch_orders: list = [{
    #     **{'pair': 'BTC_USDT'},
    #     **{'action': 'BUY'},
    #     **{'amount': str(0.0001)},
    #     **({'price': str(38000)}),
    #     **{'timestamp': get_current_timestamp()},
    #     **{'type': 'LIMIT'},
    # }, {
    #     **{'pair': 'BTC_USDT'},
    #     **{'action': 'BUY'},
    #     **{'amount': str(0.0001)},
    #     **({'price': str(38001)}),
    #     **{'timestamp': get_current_timestamp()},
    #     **{'type': 'LIMIT'},
    # }]
    batch_orders: list = [orders_1, orders_2]
    r = bitopro_client.create_batch_order(batch_orders)
    print(r)


def main():
    # ui.load_data(set_single_buy)
    ui.load_data(set_multiple_buy_sell)


if __name__ == "__main__":
    main()

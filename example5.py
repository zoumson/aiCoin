from bitoproClient.bitopro_restful_client import BitoproRestfulClient
from bitoproClient.bitopro_util import get_current_timestamp
import uicredential as ui


def get_bid_ask(_email, _api_secret, _api_key):
    # Create and initialize an instance of BitoproRestfulClient
    bitopro_client = BitoproRestfulClient(_api_key, _api_secret)

    # Set trading pair to btc_usdt
    # pair = "btc_usdt"
    pair = "sol_twd"
    # We retrieve the first level bid and ask from the order book
    orderbook = bitopro_client.get_order_book(pair=pair, limit=1, scale=0)
    print(orderbook, '\n')

    # Highest bid price
    bid = float(orderbook['bids'][0]['price']) if len(orderbook['bids']) > 0 else None

    # Lowest ask price
    ask = float(orderbook['asks'][0]['price']) if len(orderbook['asks']) > 0 else None

    # Spread
    spread = (ask - bid) if (bid and ask) else None

    # Mid price
    mid = 0.5 * (bid + ask) if (bid and ask) else None

    # Print market data
    print(f"Highest bid price: {bid:.2f}, "
          f"Lowest ask price: {ask:.2f}, "
          f"Mid price: {mid:.2f}, "
          f"Spread: {spread:.2f}")


def main():
    ui.load_data(get_bid_ask)


if __name__ == "__main__":
    main()


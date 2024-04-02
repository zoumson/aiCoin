import requests
import json
import hmac
import hashlib
import base64
import time
import tkinter as tk
from tkinter import TclError, ttk

import uicredential as ui


def send_request(method, url, headers=None, data=None, timeout=None):
    try:
        session = requests.Session()
        response = None
        if method == "GET":
            response = session.get(url, headers=headers, params=data, timeout=timeout)
        if method == "POST":
            response = session.post(url, headers=headers, json=data, timeout=timeout)
        if method == "DELETE":
            response = session.delete(url, headers=headers, timeout=timeout)

        if response is not None and response.status_code == requests.codes.ok:
            return response.json()
        else:
            return None
    except Exception as ex:
        print(ex)


def connect_now(_email, _api_secret, _api_key):
    # generate payload
    params = {"identity": _email, "nonce": int(time.time() * 1000)}

    # base64 encode to get payload
    payload = base64.urlsafe_b64encode(json.dumps(params).encode("utf-8")).decode("utf-8")

    # use api secret to get signature
    signature = hmac.new(
        bytes(_api_secret, "utf-8"),
        bytes(payload, "utf-8"),
        hashlib.sha384,
    ).hexdigest()

    # combine these data into an HTTP request header
    headers = {
        "X-BITOPRO-APIKEY": _api_key,
        "X-BITOPRO-PAYLOAD": payload,
        "X-BITOPRO-SIGNATURE": signature,
    }
    base_url = 'https://api.bitopro.com/v3'
    # combine endpoint with baseUrl
    endpoint = "/accounts/balance"
    complete_url = base_url + endpoint

    # send http request to server
    response = send_request(method="GET", url=complete_url, headers=headers)
    if response is not None:
        print("Account balance:", json.dumps(response, indent=2))
        # twd = print(response['data'][0])
        # matic = response['data'][11]
        # sol = response['data'][14]
    else:
        print("Request failed.")


# execute pip install requests first
def main():
    ui.load_data(connect_now)


if __name__ == "__main__":
    main()

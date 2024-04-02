import tkinter as tk
from tkinter import TclError, ttk


def create_input_frame(container, _api_info):
    frame = ttk.Frame(container)

    # grid layout for the input frame
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(0, weight=3)

    ttk.Label(frame, text='API Key').grid(column=0, row=0, sticky=tk.W)
    key = ttk.Entry(frame, textvariable=_api_info[0], width=30, show='*')
    key.focus()
    key.grid(column=1, row=0, sticky=tk.W)

    ttk.Label(frame, text='API Secret').grid(column=0, row=1, sticky=tk.W)
    secret = ttk.Entry(frame, textvariable=_api_info[1], width=30, show='*')
    secret.grid(column=1, row=1, sticky=tk.W)

    ttk.Label(frame, text='Email').grid(column=0, row=2, sticky=tk.W)
    mail = ttk.Entry(frame, textvariable=_api_info[2], width=30)
    mail.grid(column=1, row=2, sticky=tk.W)

    for widget in frame.winfo_children():
        widget.grid(padx=5, pady=5)

    return frame


def create_button_frame(container, _read_data_connect):
    frame = ttk.Frame(container)

    frame.columnconfigure(0, weight=1)

    ttk.Button(frame, text='Send', command=_read_data_connect).grid(column=0, row=1)

    for widget in frame.winfo_children():
        widget.grid(padx=5, pady=5)

    return frame


def create_main_window(_callback=None):
    root = tk.Tk()
    api_info = [tk.StringVar(), tk.StringVar(), tk.StringVar()]
    root.title('Replace')
    root.resizable(0, 0)
    root.geometry("600x600")
    try:
        # windows only (remove the minimize/maximize button)
        root.attributes('-toolwindow', True)
    except TclError:
        print('Not supported on your platform')

    # layout on the root window
    root.columnconfigure(0, weight=4)
    root.columnconfigure(1, weight=1)

    input_frame = create_input_frame(root, api_info)
    input_frame.grid(column=0, row=0)

    def read_data_connect():
        api_key = api_info[0].get()
        api_secret = api_info[1].get()
        email = api_info[2].get()
        _callback(email, api_secret, api_key)
        #
        # print(api_key)
        # print(api_secret)
        # print(email)

    button_frame = create_button_frame(root, read_data_connect)
    button_frame.grid(column=1, row=0)

    w = root.winfo_reqwidth()
    h = root.winfo_reqheight()
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    root.geometry('+%d+%d' % (x, y))

    root.mainloop()


def load_data(_callback=None):
    create_main_window(_callback)

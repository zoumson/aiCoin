import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pickle
from backtesting import Backtest, Strategy

global cwd
cwd = os.getcwd()


def ex11(ticker: str):
    # ticker = 'SOL-USD'
    data_frame = yf.download(ticker)
    data_frame = data_frame.drop(columns='Adj Close')
    dir = f'{cwd}/resource/'
    dir_sub1 = f'{cwd}/resource/data/'
    dir_sub2 = f'{cwd}/resource/models/'
    dir_sub3 = f'{cwd}/resource/reports/'
    os.makedirs(dir, exist_ok=True)
    os.makedirs(dir_sub1, exist_ok=True)
    os.makedirs(dir_sub2, exist_ok=True)
    os.makedirs(dir_sub3, exist_ok=True)
    data_frame.to_csv(f"{dir_sub1}Solana_Price_Historical_Daily.csv")
    print(data_frame)
    # fig = go.Figure(data=[go.Candlestick(
    #     x=data_frame.index,
    #     open=data_frame['Open'],
    #     high=data_frame['High'],
    #     low=data_frame['Low'],
    #     close=data_frame['Close'])])
    # fig.update_layout(xaxis_rangeslider_visible=False)
    # fig.show()


# count up and down on close change
def ex12():
    # cwd = os.getcwd()
    file_path = f'{cwd}/resource/data/Solana_Price_Historical_Daily.csv'
    data_frame = pd.read_csv(
        file_path,
        parse_dates=['Date'], index_col=0
    )

    data_frame_cpy = data_frame.loc['2021-03-26':, :].copy()
    # data_frame_cpy['change_tomorrow'] = data_frame_cpy.Close.pct_change(-1) * 100 * -1
    data_frame_cpy['change_tomorrow'] = data_frame_cpy.Close.pct_change(-1) * -1
    data_frame_cpy = data_frame_cpy.dropna().copy()
    data_frame_cpy['change_tomorrow_direction'] = np.where(
        data_frame_cpy.change_tomorrow > 0, 'UP', 'DOWN')

    num_up_down = data_frame_cpy.change_tomorrow_direction.value_counts()
    # data_frame_cpy.Close.plot()
    # plt.show()
    print(data_frame_cpy.head())


def ex13():
    file_path = f'{cwd}/resource/data/Solana_Price_Historical_Daily.csv'
    data_frame = pd.read_csv(
        file_path,
        parse_dates=['Date'], index_col=0
    )

    data_frame_cpy = data_frame.loc['2021-03-26':, :].copy()
    # data_frame_cpy['change_tomorrow'] = data_frame_cpy.Close.pct_change(-1) * 100 * -1
    data_frame_cpy['change_tomorrow'] = data_frame_cpy.Close.pct_change(-1) * -1
    data_frame_cpy = data_frame_cpy.dropna().copy()
    data_frame_cpy['change_tomorrow_direction'] = np.where(
        data_frame_cpy.change_tomorrow > 0, 'UP', 'DOWN')
    data_frame_cpy.to_csv(file_path)
    '''
    Separate the data
    Target: which variable do you want to predict?
    Explanatory: which variables will you use to calculate the prediction?
    '''

    target = data_frame_cpy.change_tomorrow_direction
    explanatory = data_frame_cpy[['Open', 'High', 'Low', 'Close', 'Volume']]
    model_dt = DecisionTreeClassifier(max_depth=5)
    model_dt.fit(explanatory, target)
    # print(model_dt)
    # Visualize the model
    # plot_tree(decision_tree=model_dt, feature_names=model_dt.feature_names_in_)
    # plt.show()
    # Calculate the predictions
    y_pred = model_dt.predict(X=explanatory)
    df_predictions = data_frame_cpy[['change_tomorrow_direction']].copy()
    df_predictions['prediction'] = y_pred
    print(df_predictions.head())
    # Evaluate the model: compare predictions with the reality
    comp = df_predictions.change_tomorrow_direction == df_predictions.prediction
    pred_acc = comp.sum() / len(comp)
    # print(pred_acc)
    with open(f'{cwd}/resource/models/model_dt_classification.pkl', 'wb') as f:
        pickle.dump(model_dt, f)

    """
    notebooks-course/03B_Backtesting ML Classification-Based.ipynb
    """


"""
simple backtest using the entire data used for training 
"""


def ex14():
    with open(f'{cwd}/resource/models/model_dt_classification.pkl', 'rb') as f:
        model_dt = pickle.load(f)

    file_path = f'{cwd}/resource/data/Solana_Price_Historical_Daily.csv'
    df = pd.read_csv(file_path, index_col=0, parse_dates=['Date'])
    df_explanatory = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    class SimpleClassificationUD(Strategy):
        def __init__(self, broker, data, params):
            super().__init__(broker, data, params)
            self.already_bought = None
            self.model = None

        def init(self):
            self.model = model_dt
            self.already_bought = False

        def next(self):
            explanatory_today = self.data.df.iloc[-1:, :]
            forecast_tomorrow = self.model.predict(explanatory_today)[0]

            # conditions to sell or buy
            if forecast_tomorrow == 'UP' and self.already_bought == False:
                self.buy()
                self.already_bought = True
            elif forecast_tomorrow == 'DOWN' and self.already_bought == True:
                self.sell()
                self.already_bought = False
            else:
                pass

    bt = Backtest(df_explanatory, SimpleClassificationUD,
                  cash=10000, commission=.002, exclusive_orders=True)
    results = bt.run()
    print(results.to_frame(name='Values').loc[:'Return [%]'])



def ex13_2():
    file_path = f'{cwd}/resource/data/Solana_Price_Historical_Daily.csv'
    data_frame = pd.read_csv(
        file_path,
        parse_dates=['Date'], index_col=0
    )

    data_frame_cpy = data_frame.loc['2021-03-26':, :].copy()
    # data_frame_cpy['change_tomorrow'] = data_frame_cpy.Close.pct_change(-1) * 100 * -1
    data_frame_cpy['change_tomorrow'] = data_frame_cpy.Close.pct_change(-1) * -1
    data_frame_cpy = data_frame_cpy.dropna().copy()
    data_frame_cpy['change_tomorrow_direction'] = np.where(
        data_frame_cpy.change_tomorrow > 0, 'UP', 'DOWN')
    data_frame_cpy.to_csv(file_path)
    '''
    Separate the data
    Target: which variable do you want to predict?
    Explanatory: which variables will you use to calculate the prediction?
    '''

    target = data_frame_cpy.change_tomorrow_direction
    explanatory = data_frame_cpy[['Open', 'High', 'Low', 'Close', 'Volume']]
    model_dt = DecisionTreeClassifier(max_depth=5)
    model_dt.fit(explanatory, target)
    # print(model_dt)
    # Visualize the model
    # plot_tree(decision_tree=model_dt, feature_names=model_dt.feature_names_in_)
    # plt.show()
    # Calculate the predictions
    y_pred = model_dt.predict(X=explanatory)
    df_predictions = data_frame_cpy[['change_tomorrow_direction']].copy()
    df_predictions['prediction'] = y_pred
    print(df_predictions.head())
    # Evaluate the model: compare predictions with the reality
    comp = df_predictions.change_tomorrow_direction == df_predictions.prediction
    pred_acc = comp.sum() / len(comp)
    print(pred_acc)
    # with open(f'{cwd}/resource/models/model_dt_classification.pkl', 'wb') as f:
    #     pickle.dump(model_dt, f)
    #
    # """
    # notebooks-course/03B_Backtesting ML Classification-Based.ipynb
    # """


import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
import pickle
from backtesting import Backtest, Strategy

global cwd, file_path
cwd = os.getcwd()
file_path = f'{cwd}/resource/data/Solana_Price_Historical_Daily.csv'

"""
machine learning regression model 
"""


def ex21():
    # file_path = f'{cwd}/resource/data/Solana_Price_Historical_Daily.csv'
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col=0)
    target = df.change_tomorrow
    explanatory = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    model_dt = DecisionTreeRegressor(max_depth=5)
    model_dt.fit(explanatory, target)
    y_pred = model_dt.predict(X=explanatory)
    df_predictions = df[['change_tomorrow']].copy()
    df_predictions['prediction'] = y_pred
    error = df_predictions.change_tomorrow - df_predictions.prediction
    error_squared = error ** 2
    error_squared_mean = error_squared.mean()
    rmse = np.sqrt(error_squared_mean)
    error.std()
    error.hist(bins=30)
    plt.show()

    with open(f'{cwd}/resource/models/model_dt_regression.pkl', 'wb') as f:
        pickle.dump(model_dt, f)


def ex22():
    with open(f'{cwd}/resource/models/model_dt_regression.pkl', 'rb') as f:
        model_dt = pickle.load(f)

    df = pd.read_csv(file_path, index_col=0, parse_dates=['Date'])

    class Regression(Strategy):
        def __init__(self, broker, data, params):
            super().__init__(broker, data, params)
            self.already_bought = None
            self.model = None

        def init(self):
            self.model = model_dt
            self.already_bought = False

        def next(self):
            explanatory_today = self.data.df.iloc[[-1], :]
            forecast_tomorrow = self.model.predict(explanatory_today)[0]

            if forecast_tomorrow > 1 and self.already_bought == False:
                self.buy()
                self.already_bought = True
            elif forecast_tomorrow < -5 and self.already_bought == True:
                self.sell()
                self.already_bought = False
            else:
                pass

    # Define initial conditions
    df_explanatory = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()


    # Run backtesting
    bt = Backtest(df_explanatory, Regression,
                  cash=10000, commission=.002, exclusive_orders=True)
    results = bt.run()
    # Interpret backtesting results
    results_table = results.to_frame(name='Values').loc[:'Return [%]']
    print(results_table)

    bt.plot(filename=f'{cwd}/resource/reports/backtesting_regression.html')
    plt.show()


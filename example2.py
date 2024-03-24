import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
import pickle
from backtesting import Backtest, Strategy

global cwd
cwd = os.getcwd()


"""
machine learning regression model 
"""
def ex21():

    file_path = f'{cwd}/resource/data/Solana_Price_Historical_Daily.csv'
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

    with open(f'{cwd}/resource/models/model_dt_regression.pkl', 'wb') as f:
        pickle.dump(model_dt, f)





    
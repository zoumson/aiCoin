import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, SGDOneClassSVM
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor

import pickle
from backtesting import Backtest, Strategy
import multiprocessing as mp
from skopt.plots import plot_evaluations, plot_objective
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from strategies import WalkForwardAnchored, RegressionCustomModFixLimitBuySell

global cwd, file_path
cwd = os.getcwd()


def ex41(ticker: str):
    # Download data from yfinance
    df = yf.download(ticker)

    """
    Preprocess the data
    Filter the date range
    """
    df = df.loc['2022-01-01':].copy()
    """
    Create the target variable
    """
    df['change_tomorrow'] = df['Adj Close'].pct_change(-1)
    df.change_tomorrow = df.change_tomorrow * -1
    df.change_tomorrow = df.change_tomorrow * 100

    # Remove rows with any missing data
    df = df.dropna().copy()

    """
    Machine Learning modelling
    """
    # Feature selection
    # Target: which variable do you want to predict?
    # Explanatory: which variables will you use to calculate the prediction?
    y = df.change_tomorrow
    # All the columns except the target column which is the
    # the change in percentage that will occur the next day
    x = df.drop(columns='change_tomorrow')

    # Walk Forward Validation
    # a model is created and trained then test repeatedly along on
    # the simulation period
    ts = TimeSeriesSplit(max_train_size=200, gap=3, test_size=100)
    model1 = GradientBoostingRegressor()
    model2 = HistGradientBoostingRegressor()
    model3 = RandomForestRegressor()
    model4 = ExtraTreesRegressor()
    model5 = AdaBoostRegressor()
    model6 = BaggingRegressor()

    r1 = LinearRegression()
    r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    r3 = KNeighborsRegressor()

    model7 = VotingRegressor([('lr', r1), ('rf', r2), ('r3', r3)])

    estimators = [('lr', RidgeCV()), ('svr', LinearSVR(dual="auto", random_state=42))]
    model8 = StackingRegressor(
        estimators=estimators,
        final_estimator=RandomForestRegressor(n_estimators=10, random_state=42))

    model9 = MLPRegressor(random_state=1, max_iter=10)

    error_mse_list1 = []
    error_mse_list2 = []
    error_mse_list3 = []
    error_mse_list4 = []
    error_mse_list5 = []
    error_mse_list6 = []
    error_mse_list7 = []
    error_mse_list8 = []
    error_mse_list9 = []

    for index_train, index_test in ts.split(df):
        x_train, y_train = x.iloc[index_train], y.iloc[index_train]
        x_test, y_test = x.iloc[index_test], y.iloc[index_test]

        model1.fit(x_train, y_train)
        model2.fit(x_train, y_train)
        model3.fit(x_train, y_train)
        model4.fit(x_train, y_train)
        model5.fit(x_train, y_train)
        model6.fit(x_train, y_train)
        model7.fit(x_train, y_train)
        model8.fit(x_train, y_train)
        model9.fit(x_train, y_train)

        y_pred1 = model1.predict(x_test)
        y_pred2 = model2.predict(x_test)
        y_pred3 = model3.predict(x_test)
        y_pred4 = model4.predict(x_test)
        y_pred5 = model5.predict(x_test)
        y_pred6 = model6.predict(x_test)
        y_pred7 = model7.predict(x_test)
        y_pred8 = model8.predict(x_test)
        y_pred9 = model9.predict(x_test)

        error_mse1 = mean_squared_error(y_test, y_pred1)
        error_mse2 = mean_squared_error(y_test, y_pred2)
        error_mse3 = mean_squared_error(y_test, y_pred3)
        error_mse4 = mean_squared_error(y_test, y_pred4)
        error_mse5 = mean_squared_error(y_test, y_pred5)
        error_mse6 = mean_squared_error(y_test, y_pred6)
        error_mse7 = mean_squared_error(y_test, y_pred7)
        error_mse8 = mean_squared_error(y_test, y_pred8)
        error_mse9 = mean_squared_error(y_test, y_pred9)

        error_mse_list1.append(error_mse1)
        error_mse_list2.append(error_mse2)
        error_mse_list3.append(error_mse3)
        error_mse_list4.append(error_mse4)
        error_mse_list5.append(error_mse5)
        error_mse_list6.append(error_mse6)
        error_mse_list7.append(error_mse7)
        error_mse_list8.append(error_mse8)
        error_mse_list9.append(error_mse9)

    print(f"GradientBoostingRegressor: {np.mean(error_mse_list1)}")
    print(f"HistGradientBoostingRegressor: {np.mean(error_mse_list2)}")
    print(f"RandomForestRegressor: {np.mean(error_mse_list3)}")
    print(f"ExtraTreesRegressor: {np.mean(error_mse_list4)}")
    print(f"AdaBoostRegressor: {np.mean(error_mse_list5)}")
    print(f"BaggingRegressor: {np.mean(error_mse_list6)}")
    print(f"VotingRegressor: {np.mean(error_mse_list7)}")
    print(f"StackingRegressor: {np.mean(error_mse_list8)}")
    print(f"MLPRegressor: {np.mean(error_mse_list9)}")

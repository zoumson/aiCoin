import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, SGDOneClassSVM
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

import pickle
from backtesting import Backtest, Strategy
import multiprocessing as mp
from skopt.plots import plot_evaluations, plot_objective
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from strategies import WalkForwardAnchored, RegressionCustomModFixLimitBuySell

global cwd, file_path
cwd = os.getcwd()
file_path = f'{cwd}/resource/data/Solana_Price_Historical_Daily.csv'


def ex31(ticker: str):
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
    # splits = ts.split(X=df)

    # list_df_train = []
    # list_df_test = []
    #
    # for index_train, index_test in ts.split(df):
    #     x_train, y_train = x.iloc[index_train], y.iloc[index_train]
    #     X_test, y_test = x.iloc[index_test], y.iloc[index_test]

    model_dt = DecisionTreeRegressor(max_depth=5, random_state=10)

    error_mse_list = []

    for index_train, index_test in ts.split(df):
        x_train, y_train = x.iloc[index_train], y.iloc[index_train]
        x_test, y_test = x.iloc[index_test], y.iloc[index_test]

        model_dt.fit(x_train, y_train)

        y_pred = model_dt.predict(x_test)
        error_mse = mean_squared_error(y_test, y_pred)
        error_mse_list.append(error_mse)
    model_path = f'{cwd}/resource/models/model_dt_regression_1.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_dt, f)

    # bt = Backtest(df, RegressionCustomModFixLimitBuySell(model_path), cash=2000, commission=.002, exclusive_orders=True)
    # results = bt.run(limit_buy=1, limit_sell=-5)
    # #
    # df_results = results.to_frame(name='Values').loc[:'Return [%]'] \
    #     .rename({'Values': 'Out of Sample'}, axis=1)
    #
    # print(df_results)
    # print(f"error : {np.mean(error_mse_list)}")
    # x_today = x.iloc[[-1]]
    # y_tomorrow = model_dt.predict(x_today)
    # # print(x.iloc[[-1]])
    # print(f"today : {x_today}")
    # print(f"tomorrow : {y_tomorrow}")
    # class Regression(Strategy):
    #     limit_buy = 1
    #     limit_sell = -5
    #
    #     n_train = 600
    #     coef_retrain = 200
    #
    #     def __init__(self, broker, data, params):
    #         super().__init__(broker, data, params)
    #         self.model = None
    #         self.already_bought = None
    #
    #     def init(self):
    #         self.model = DecisionTreeRegressor(max_depth=5, random_state=10)
    #         self.already_bought = False
    #
    #         x_train = self.data.df.iloc[:self.n_train, :-1]
    #         y_train = self.data.df.iloc[:self.n_train, -1]
    #
    #         self.model.fit(X=x_train, y=y_train)
    #
    #     def next(self):
    #         explanatory_today = self.data.df.iloc[[-1], :-1]
    #         forecast_tomorrow = self.model.predict(explanatory_today)[0]
    #
    #         if forecast_tomorrow > self.limit_buy and self.already_bought == False:
    #             self.buy()
    #             self.already_bought = True
    #         elif forecast_tomorrow < self.limit_sell and self.already_bought == True:
    #             self.sell()
    #             self.already_bought = False
    #         else:
    #             pass
    #
    # class WalkForwardAnchored(Regression):
    #     def next(self):
    #         # we don't take any action and move on to the following day
    #         if len(self.data) < self.n_train:
    #             return
    #
    #         # we retrain the model each 200 days
    #         if len(self.data) % self.coef_retrain == 0:
    #             x_train = self.data.df.iloc[:, :-1]
    #             y_train = self.data.df.iloc[:, -1]
    #
    #             self.model.fit(x_train, y_train)
    #             super().next()
    #         else:
    #             super().next()
    #
    # bt = Backtest(df, WalkForwardAnchored, cash=2000, commission=.002, exclusive_orders=True)
    # results = bt.run(limit_buy=1, limit_sell=-5)
    #
    # df_results_ = results.to_frame(name='Values').loc[:'Return [%]'] \
    #     .rename({'Values': 'Out of Sample (Test)'}, axis=1)
    #
    # print(bt)
    # stats_skopt, heatmap, optimize_result = bt.optimize(
    #     limit_buy=range(0, 6), limit_sell=range(-6, 0),
    #     maximize='Return [%]',
    #     max_tries=500,
    #     random_state=42,
    #     return_heatmap=True,
    #     return_optimization=True,
    #     method='skopt'
    # )
    #
    # dff = heatmap.reset_index()
    # dff = dff.sort_values('Return [%]', ascending=False)
    # print(dff)
    #
    # bt_unanchored = Backtest(df, strategies.WalkForwardUnanchored, cash=10000, commission=.002, exclusive_orders=True)
    #
    # stats_skopt, heatmap, optimize_result = bt_unanchored.optimize(
    #     limit_buy=range(0, 6), limit_sell=range(-6, 0),
    #     maximize='Return [%]',
    #     max_tries=20,
    #     random_state=10,
    #     return_heatmap=True,
    #     return_optimization=True,
    #     method='skopt'
    # )
    #
    # dff = heatmap.reset_index()
    # dff = dff.sort_values('Return [%]', ascending=False)
    # print(dff)
    # bt.plot(filename=f'{cwd}/resource/reports/walk_forward_anchored.html')
    # plt.show()
    # bt_unanchored.plot(filename=f'{cwd}/resource/reports/walk_forward_unanchored.html')
    # plt.show()


def ex32(ticker: str):
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
    model1 = LinearRegression()
    model2 = Ridge()
    model3 = Lasso()
    model4 = ElasticNet(random_state=0)
    model5 = SGDRegressor()
    model6 = SGDOneClassSVM()

    error_mse_list1 = []
    error_mse_list2 = []
    error_mse_list3 = []
    error_mse_list4 = []
    error_mse_list5 = []
    error_mse_list6 = []

    for index_train, index_test in ts.split(df):
        x_train, y_train = x.iloc[index_train], y.iloc[index_train]
        x_test, y_test = x.iloc[index_test], y.iloc[index_test]

        model1.fit(x_train, y_train)
        model2.fit(x_train, y_train)
        model3.fit(x_train, y_train)
        model4.fit(x_train, y_train)
        model5.fit(x_train, y_train)
        model6.fit(x_train, y_train)

        y_pred1 = model1.predict(x_test)
        y_pred2 = model2.predict(x_test)
        y_pred3 = model3.predict(x_test)
        y_pred4 = model4.predict(x_test)
        y_pred5 = model5.predict(x_test)
        y_pred6 = model6.predict(x_test)
        error_mse1 = mean_squared_error(y_test, y_pred1)
        error_mse2 = mean_squared_error(y_test, y_pred2)
        error_mse3 = mean_squared_error(y_test, y_pred3)
        error_mse4 = mean_squared_error(y_test, y_pred4)
        error_mse5 = mean_squared_error(y_test, y_pred5)
        error_mse6 = mean_squared_error(y_test, y_pred6)
        error_mse_list1.append(error_mse1)
        error_mse_list2.append(error_mse2)
        error_mse_list3.append(error_mse3)
        error_mse_list4.append(error_mse4)
        error_mse_list5.append(error_mse5)
        error_mse_list6.append(error_mse6)

    print(f"Linear: {np.mean(error_mse_list1)}")
    print(f"Ridge: {np.mean(error_mse_list2)}")
    print(f"Lasso: {np.mean(error_mse_list3)}")
    print(f"Elastic Net: {np.mean(error_mse_list4)}")
    print(f"SGDRegressor: {np.mean(error_mse_list5)}")
    print(f"SGDOneClassSVM: {np.mean(error_mse_list6)}")
    # Lasso: 27.64714727329578


def ex33(ticker: str):
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
    model1 = SVR()

    error_mse_list1 = []

    for index_train, index_test in ts.split(df):
        x_train, y_train = x.iloc[index_train], y.iloc[index_train]
        x_test, y_test = x.iloc[index_test], y.iloc[index_test]

        model1.fit(x_train, y_train)

        y_pred1 = model1.predict(x_test)
        error_mse1 = mean_squared_error(y_test, y_pred1)
        error_mse_list1.append(error_mse1)

    print(f"SVM: {np.mean(error_mse_list1)}")
    # Lasso: 27.64714727329578
    # SVM: 23.568902356724
    # SGDOneClassSVM: 23.509953009388205


def ex34(ticker: str):
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
    model1 = SVR()
    model2 = KNeighborsRegressor(n_neighbors=50)
    # model3 = GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=0)
    model3 = GaussianProcessRegressor()

    error_mse_list1 = []
    error_mse_list2 = []
    error_mse_list3 = []

    for index_train, index_test in ts.split(df):
        x_train, y_train = x.iloc[index_train], y.iloc[index_train]
        x_test, y_test = x.iloc[index_test], y.iloc[index_test]

        model1.fit(x_train, y_train)
        model2.fit(x_train, y_train)
        model3.fit(x_train, y_train)

        y_pred1 = model1.predict(x_test)
        y_pred2 = model2.predict(x_test)
        y_pred3 = model3.predict(x_test)
        error_mse1 = mean_squared_error(y_test, y_pred1)
        error_mse2 = mean_squared_error(y_test, y_pred2)
        error_mse3 = mean_squared_error(y_test, y_pred3)
        error_mse_list1.append(error_mse1)
        error_mse_list2.append(error_mse2)
        error_mse_list3.append(error_mse3)

    print(f"SVM: {np.mean(error_mse_list1)}")
    print(f"KNeighborsRegressor: {np.mean(error_mse_list2)}")
    print(f"GaussianProcessRegressor: {np.mean(error_mse_list3)}")
    # Lasso: 27.64714727329578
    # SVM: 23.568902356724
    # KNeighborsRegressor: 23.513686845532213
    # GaussianProcessRegressor: 23.298421754743003

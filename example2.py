import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
import pickle
from backtesting import Backtest, Strategy
import multiprocessing as mp
from skopt.plots import plot_evaluations, plot_objective
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import strategies



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


def ex23():
    # Load the model
    with open(f'{cwd}/resource/models/model_dt_regression.pkl', 'rb') as f:
        model_dt = pickle.load(f)

    df = pd.read_csv(file_path, parse_dates=['Date'], index_col=0)

    class Regression(Strategy):

        limit_buy = 1
        limit_sell = -5

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

            if forecast_tomorrow > self.limit_buy and self.already_bought == False:
                self.buy()
                self.already_bought = True
            elif forecast_tomorrow < self.limit_sell and self.already_bought == True:
                self.sell()
                self.already_bought = False
            else:
                pass

    df_explanatory = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    bt = Backtest(df_explanatory, Regression,
                  cash=10000, commission=.002, exclusive_orders=True)

    list_limits_buy = list(range(0, 5, 2))
    list_limits_sell = list(range(0, -5, -2))

    mp.set_start_method('fork')
    stats_skopt, heatmap, optimize_result = bt.optimize(
        limit_buy=[0, 5],
        limit_sell=[-5, 0],
        maximize='Return [%]',
        method='skopt',
        max_tries=500,
        random_state=0,
        return_heatmap=True,
        return_optimization=True)

    # Interpret backtesting results
    df_results_heatmap = heatmap.reset_index()
    dff = df_results_heatmap.pivot(
        index='limit_buy', columns='limit_sell', values='Return [%]')

    # dff.sort_index(axis=1, ascending=False)
    # dff.sort_index(axis=1, ascending=False).style.format(precision=0).background_gradient()
    dff.sort_index(axis=1, ascending=False).style.format(precision=0).background_gradient(vmin=np.nanmin(dff),
                                                                                          vmax=np.nanmax(dff))

    print(dff)
    _ = plot_evaluations(optimize_result, bins=10)
    plt.show()

    _ = plot_objective(optimize_result, n_points=10)
    plt.show()


def ex24(ticker: str):
    # Download data from yfinance
    df = yf.download(ticker)

    """
    Preprocess the data
    Filter the date range
    """
    df = df.loc['2021-01-01':].copy()
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
    x = df.drop(columns='change_tomorrow')

    n_days = len(df.index)
    n_days_split = int(n_days * 0.7)

    x_train, y_train = x.iloc[:n_days_split], y.iloc[:n_days_split]
    x_test, y_test = x.iloc[n_days_split:], y.iloc[n_days_split:]

    # Fit the model on train set
    model_dt_split = DecisionTreeRegressor(max_depth=5, random_state=10)
    model_dt_split.fit(X=x_train, y=y_train)

    # Evaluate model
    y_pred_test = model_dt_split.predict(X=x_test)
    error_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
    print(error_test)

    y_pred_train = model_dt_split.predict(X=x_train)
    error_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)
    print(error_train)

    # Backtesting
    class Regression(Strategy):
        limit_buy = 1
        limit_sell = -5

        def __init__(self, broker, data, params):
            super().__init__(broker, data, params)
            self.already_bought = None
            self.model = None

        def init(self):
            self.model = DecisionTreeRegressor(max_depth=5, random_state=10)
            self.already_bought = False

            self.model.fit(X=x_train, y=y_train)

        def next(self):
            explanatory_today = self.data.df.iloc[[-1], :]
            forecast_tomorrow = self.model.predict(explanatory_today)[0]

            if forecast_tomorrow > self.limit_buy and self.already_bought == False:
                self.buy()
                self.already_bought = True
            elif forecast_tomorrow < self.limit_sell and self.already_bought == True:
                self.sell()
                self.already_bought = False
            else:
                pass

    # Run the backtest on test data
    bt_test = Backtest(x_test, Regression,
                       cash=10000, commission=.002, exclusive_orders=True)
    results = bt_test.run(limit_buy=1, limit_sell=-5)

    df_results_test = results.to_frame(name='Values').loc[:'Return [%]'] \
        .rename({'Values': 'Out of Sample (Test)'}, axis=1)
    # Run the backtest on train data
    bt_train = Backtest(x_train, Regression,
                        cash=10000, commission=.002, exclusive_orders=True)

    results = bt_train.run(limit_buy=1, limit_sell=-5)

    df_results_train = results.to_frame(name='Values').loc[:'Return [%]'] \
        .rename({'Values': 'In Sample (Train)'}, axis=1)

    df_results = pd.concat([df_results_train, df_results_test], axis=1)

    bt_test.plot(filename=f'{cwd}/resource/reports/regression_test_set.html')
    plt.show()
    bt_train.plot(filename=f'{cwd}/resource/reports/regression_train_set.html')
    plt.show()


"""
Walk Forward: A Realistic Approach to Backtesting
"""


def ex25(ticker: str):
    # Download data from yfinance
    df = yf.download(ticker)

    """
    Preprocess the data
    Filter the date range
    """
    df = df.loc['2021-01-01':].copy()
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
    x = df.drop(columns='change_tomorrow')

    # Walk Forward Validation
    ts = TimeSeriesSplit(test_size=200)
    # splits = ts.split(X=df)

    list_df_train = []
    list_df_test = []

    for index_train, index_test in ts.split(df):
        list_df_train.append(df.iloc[index_train])
        list_df_test.append(df.iloc[index_test])

    class Regression(Strategy):
        limit_buy = 1
        limit_sell = -5

        n_train = 600
        coef_retrain = 200

        def __init__(self, broker, data, params):
            super().__init__(broker, data, params)
            self.model = None
            self.already_bought = None

        def init(self):
            self.model = DecisionTreeRegressor(max_depth=5, random_state=10)
            self.already_bought = False

            x_train = self.data.df.iloc[:self.n_train, :-1]
            y_train = self.data.df.iloc[:self.n_train, -1]

            self.model.fit(X=x_train, y=y_train)

        def next(self):
            explanatory_today = self.data.df.iloc[[-1], :-1]
            forecast_tomorrow = self.model.predict(explanatory_today)[0]

            if forecast_tomorrow > self.limit_buy and self.already_bought == False:
                self.buy()
                self.already_bought = True
            elif forecast_tomorrow < self.limit_sell and self.already_bought == True:
                self.sell()
                self.already_bought = False
            else:
                pass

    class WalkForwardAnchored(Regression):
        def next(self):
            # we don't take any action and move on to the following day
            if len(self.data) < self.n_train:
                return

            # we retrain the model each 200 days
            if len(self.data) % self.coef_retrain == 0:
                x_train = self.data.df.iloc[:, :-1]
                y_train = self.data.df.iloc[:, -1]

                self.model.fit(x_train, y_train)
                super().next()
            else:
                super().next()

    bt = Backtest(df, WalkForwardAnchored, cash=10000, commission=.002, exclusive_orders=True)

    stats_skopt, heatmap, optimize_result = bt.optimize(
        limit_buy=range(0, 6), limit_sell=range(-6, 0),
        maximize='Return [%]',
        max_tries=500,
        random_state=42,
        return_heatmap=True,
        return_optimization=True,
        method='skopt'
    )

    dff = heatmap.reset_index()
    dff = dff.sort_values('Return [%]', ascending=False)
    print(dff)

    bt_unanchored = Backtest(df, strategies.WalkForwardUnanchored, cash=10000, commission=.002, exclusive_orders=True)

    stats_skopt, heatmap, optimize_result = bt_unanchored.optimize(
        limit_buy=range(0, 6), limit_sell=range(-6, 0),
        maximize='Return [%]',
        max_tries=20,
        random_state=10,
        return_heatmap=True,
        return_optimization=True,
        method='skopt'
    )

    dff = heatmap.reset_index()
    dff = dff.sort_values('Return [%]', ascending=False)
    print(dff)
    bt.plot(filename=f'{cwd}/resource/reports/walk_forward_anchored.html')
    plt.show()
    bt_unanchored.plot(filename=f'{cwd}/resource/reports/walk_forward_unanchored.html')
    plt.show()
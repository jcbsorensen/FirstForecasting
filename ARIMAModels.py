from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import timedelta
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
import warnings


# split a dataset into train/test sets
def MS_split_dataset(dataframe, config):
    multi_steps, repeats, target = config
    test_slice = multi_steps * repeats
    # split into standard weeks
    train, test = dataframe[:-test_slice], dataframe[-test_slice:]
    # restructure into windows of weekly data
    train = train.iloc[:, 0]
    test = test.iloc[:, 0]
    return train, test


# evaluate one or more weekly forecasts against expected values
def MS_evaluate_forecasts(df, multi_steps, repeats):
    # calculate an RMSE score for each day
    df.loc[:, 'rmse'] = 0
    df.loc[:, 'squared error'] = 0
    for i in range(multi_steps):
        df.iloc[i::multi_steps, 2] = np.sqrt(mean_squared_error(df.iloc[i::multi_steps, 0], df.iloc[i::multi_steps, 1]))
        df.iloc[i::multi_steps, 3] = (df.iloc[i::multi_steps, 0] - df.iloc[i::multi_steps, 1]) ** 2
    # calculate overall RMSE
    errorSum = df['squared error'].sum()
    score = np.sqrt(errorSum / (repeats * multi_steps))
    scores = df.iloc[0:multi_steps, 2]
    return round(score, 2), scores


# evaluate a single model
def MS_evaluate_model(model_config, train, test, setting):
    multi_steps, repeats, target = setting
    model, order, sorder, trend = model_config
    # history is a list of weekly data
    history = train.copy()
    # create predicted column
    test = test.to_frame(name='actual')
    test.loc[:, 'predicted'] = 0
    # walk-forward validation over each week
    for i in range(repeats):
        start = i * multi_steps
        end = (i + 1) * multi_steps
        # predict the week
        test.iloc[start:end] = model(history=history, test=test.iloc[start:end], order=order, sorder=sorder,
                                     trend=trend)
        # get real observation and add to history for predicting the next week
        history = history.append(test.iloc[start:end, 0])
    # evaluate predictions days for each week
    score, scores = MS_evaluate_forecasts(df=test, multi_steps=multi_steps, repeats=repeats)
    return score, scores


def MS_arima_forecast(history, test, order, sorder, trend):
    df = test.copy()
    multi_steps = df.shape[0] - 1
    # define the model
    model = ARIMA(history, order=order, freq='D')
    # fit the model
    model_fit = model.fit(disp=False)
    # make forecast
    values = model_fit.predict(history.shape[0], history.shape[0] + multi_steps)
    df.loc[:, 'predicted'] = values
    return df


# one-step sarima forecast
def MS_sarima_forecast(history, test, order, sorder, trend):
    df = test.copy()
    multi_steps = df.shape[0] - 1
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False, freq='D')
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    values = model_fit.predict(len(history), len(history) + multi_steps)
    df.loc[:, 'predicted'] = values
    return df


# create a set of sarima configs to try
def generate_configs(model,
                     seasonal=[0],
                     order=[[0, 1, 2], [0, 1], [0, 1, 2]],
                     sorder=[[0, 1, 2], [0], [0, 1, 2]],
                     trend=['n', 'c', 't', 'ct']):
    models = list()
    p_params, d_params, q_params = order
    P_params, D_params, Q_params = sorder
    t_params = trend
    # define config lists
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [model, [p, d, q], [P, D, Q, m], t]
                                    models.append(cfg)
    return models


def MS_grid_search(model_configs, train, test, setting):
    print('total_iterations: {}'.format(len(model_configs)))

    executor = Parallel(n_jobs=cpu_count(), verbose=10)
    tasks = (
    delayed(MS_evaluate_model)(model_config=[model, order, sorder, trend], train=train, test=test, setting=setting) for
    model, order, sorder, trend in model_configs)
    result = executor(tasks)

    collection = [[model_configs[i][1][0], model_configs[i][1][1], model_configs[i][1][2], model_configs[i][2][0],
                   model_configs[i][2][1], model_configs[i][2][2], model_configs[i][2][3], model_configs[i][3],
                   result[i][0]] for i in range(len(model_configs))]

    df = pd.DataFrame(collection,
                      columns=['order_p', 'order_d', 'order_q', 'sorder_P', 'sorder_D', 'sorder_Q', 'sorder_M', 'trend',
                               'result'])
    return df

## load the new file
# dataset = read_csv('E:/Deep Learning Mastery/Deep Learning Time Series Forecasting/code/chapter_18/household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
## split into train and test
# setting = [7, 46, 'lala']
# cfg = [MS_arima_forecast, [2, 0, 0], [2, 0, 0], 'n']
# train, test = MS_split_dataset(dataframe=dataset, config=setting)
# result, results = MS_evaluate_model(model_config=cfg, train=train, test=test, setting=setting)
# print(result)
##results.head(7)




#setting = [37, 5, 'Sales']
#
## ARIMA Model Grid Search
#train, test = AM.MS_split_dataset(dataframe=subset_data, config=setting)
#model_configs = AM.generate_configs(model=AM.MS_sarima_forecast, seasonal=[7], order=[[14], [0], [7]], sorder=[[9], [0], [12]], trend='n')
##result, results = AM.MS_evaluate_model(model_config=[AM.MS_sarima_forecast, [7, 0, 0], [2, 0, 1, 7], 't'], train=train, test=test, setting=setting)
#gs_df = AM.MS_grid_search(model_configs=model_configs, train=train, test=test, setting=setting)
#gs_df2 = gs_df.sort_values(by=['result']).head(14)
# RMSE: 2272

#train = subset_data.iloc[:-37, 0]
#test = subset_data.iloc[-37:,0]
#test = test.to_frame(name='actual')
#test.loc[:, 'predicted'] = 0
#test = AM.MS_sarima_forecast(history=train, test=test, order=[7, 0, 0], sorder=[2, 0, 1, 7], trend='t' )

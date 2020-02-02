# naive forecast strategies for the power usage dataset
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import timedelta
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed



# split a univariate dataset into train/test sets
def MS_split_dataset(dataframe, config):
    multi_steps, repeats, target = config
    test_slice = multi_steps * repeats
    # split into standard weeks
    train, test = dataframe[:-test_slice], dataframe[-test_slice:]
    # restructure into windows of weekly data
    train = train.loc[:, target]
    test = test.loc[:, target]
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


# previous persistence model
def MS_previous_persistence(history, test, n, offset):
    # get the total active power for the last timestep
    df = test.copy()
    value = history[-1]
    df.loc[:, 'predicted'] = value
    return df


# sequence persistence model
def MS_sequence_persistence(history, test, n, offset):
    # get the data for the prior sequence of values
    df = test.copy()
    x_days = test.shape[0]
    previous_dates = df.index - timedelta(days=x_days)
    values = history[previous_dates].values.tolist()
    df.loc[:, 'predicted'] = values
    return df


# month persistence model
def MS_x_persistence(history, test, n, offset):
    # get the data for the prior sequence of values
    df = test.copy()
    previous_dates = df.index - timedelta(days=offset)
    values = history[previous_dates].values.tolist()
    df.loc[:, 'predicted'] = values
    return df


# mean model
def MS_mean(history, test, n, offset):
    # get the data for the prior sequence of values
    df = test.copy()
    multi_steps = df.shape[0]
    if offset > 1:
        for i in range(multi_steps):
            values = list()
            history_values = history.values.tolist()
            for y in range(1, n + 1):
                yo = y * offset
                values.append(history_values[-yo])
            mean = np.mean(values)
            history = history.append(pd.Series(data=mean, index=[i]))
    else:
        for i in range(multi_steps):
            mean = np.mean(history[-n:].values.tolist())
            history = history.append(pd.Series(data=mean, index=[i]))
    values = history[-multi_steps:].values.tolist()
    df.loc[:, 'predicted'] = values
    return df


# mean model
def MS_median(history, test, n, offset):
    # get the data for the prior sequence of values
    df = test.copy()
    multi_steps = df.shape[0]
    if offset > 1:
        for i in range(multi_steps):
            values = list()
            history_values = history.values.tolist()
            for y in range(1, n + 1):
                yo = y * offset
                values.append(history_values[-yo])
            median = np.median(values)
            history = history.append(pd.Series(data=median, index=[i]))
    else:
        for i in range(multi_steps):
            median = np.median(history[-n:].values.tolist())
            history = history.append(pd.Series(data=median, index=[i]))
    values = history[-multi_steps:].values.tolist()
    df.loc[:, 'predicted'] = values
    return df


# evaluate a single model
def MS_evaluate_model(model_config, train, test, setting):
    multi_steps, repeats, target = setting
    model, n, offset = model_config
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
        test.iloc[start:end] = model(history=history, test=test.iloc[start:end], n=n, offset=offset)
        # get real observation and add to history for predicting the next week
        history = history.append(test.iloc[start:end, 0])
    # evaluate predictions days for each week
    score, scores = MS_evaluate_forecasts(df=test, multi_steps=multi_steps, repeats=repeats)
    return score, scores, test


## OLD create a set of simple configs to try
#def generate_configs(multi_steps, max_length, offsets=[1]):
#    configs = dict()
#    for i in range(1, max_length+1):
#        for o in offsets:
#            if (i * o) < max_length:
#                for t in [MS_mean, MS_median]:
#                    setting = 'model: {}; n: {}; offset: {}'.format(str(t), i, o)
#                    cfg = [t, i, o]
#                    configs[setting] = cfg
#
#                if o > multi_steps:
#                    setting = 'model: {}; n: {}; offset: {}'.format(MS_x_persistence, i, o)
#                    cfg = [MS_x_persistence, i, o]
#                    configs[setting] = cfg
#
#    configs['model: previous_persistence; n: NA; offset: NA'] = [MS_previous_persistence, 1, 1]
#    configs['model: sequence_persistence; n: NA; offset: NA'] = [MS_sequence_persistence, 1, 1]
#    return configs


# create a set of simple configs to try
def generate_configs(multi_steps, max_length, offsets=[1]):
    configs = list()
    for i in range(1, max_length+1):
        for o in offsets:
            if (i * o) < max_length:
                for t in [MS_mean, MS_median]:
                    setting = 'model: {}; n: {}; offset: {}'.format(str(t), i, o)
                    configs.append([setting, t, i, o])

                if o > multi_steps:
                    setting = 'model: {}; n: {}; offset: {}'.format(MS_x_persistence, i, o)
                    configs.append([setting, MS_x_persistence, i, o])

    configs.append(['model: previous_persistence; n: NA; offset: NA', MS_previous_persistence, 1, 1])
    configs.append(['model: sequence_persistence; n: NA; offset: NA', MS_sequence_persistence, 1, 1])
    return configs


# summarize scores
def MS_summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores.values.tolist()])
    print('%s: [%.3f] %s' % (name, score, s_scores))


## OLD grid search
#def MS_grid_search(model_configs, train, test, setting, parallel=True):
#    total_iterations = len(model_configs)
#    progress = 0
#    collection = list()
#    for name, func in model_configs.items():
#        result, results = MS_evaluate_model(model_config=func, train=train, test=test, setting=setting)
#        model, n, offset = func
#        collection.append([name, n, offset, result])
#        progress += 1
#        print('{} out of {}'.format(progress, total_iterations))
#
#    df = pd.DataFrame(collection, columns=['model', 'n', 'offset', 'result'])
#    return df


def MS_grid_search(model_configs, train, test, setting, parallel=True):
    print('total_iterations: {}'.format(len(model_configs)))
    collection = list()
    executor = Parallel(n_jobs=cpu_count(), verbose=10)
    tasks = (delayed(MS_evaluate_model)(model_config=[model, n, offset], train=train, test=test, setting=setting) for name, model, n, offset in model_configs)
    result = executor(tasks)
    #for i in range(len(model_configs)):
    #    collection.append([model_configs[i][1], model_configs[i][2], model_configs[i][3], result[i][0]])

    collection = [[model_configs[i][1], model_configs[i][2], model_configs[i][3], result[i][0]] for i in range(len(model_configs))]

    df = pd.DataFrame(collection, columns=['model', 'n', 'offset', 'result'])
    return df




#setting = [7, 5, 'Sales']

# # Naive Model Grid Search
#train, test = NM.MS_split_dataset(dataframe=subset_data, config=setting)
#model_configs = NM.generate_configs(multi_steps=setting[0], max_length=300, offsets=[1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30, 60, 90, 180, 183])
#gs_df = NM.MS_grid_search(model_configs=model_configs, train=train, test=test, setting=setting)
#result, results = NM.MS_evaluate_model(model_config=[NM.MS_median, 3, 14], train=train, test=test, setting=setting)
#df = NM.MS_median(history=subset_data.iloc[:-7, 0], test=subset_data.iloc[-7:,0], n=3, offset=14)
#gs_df = gs_df.sort_values(by=['result']).head(20)

#train = subset_data.iloc[:-7, 0]
#test = subset_data.iloc[-7:,0]
#test = test.to_frame(name='actual')
#test.loc[:, 'predicted'] = 0
#test = NM.MS_median(history=subset_data.iloc[:-7, 0], test=test, n=3, offset=14)
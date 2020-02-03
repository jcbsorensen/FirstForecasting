# univariate multi-step cnn for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import SGD
from keras.layers import Dropout
from clr_callback import *
from timeit import default_timer as timer


def MS_create_setting(multi_steps, repeats, target, preprocessing=None):
    return [multi_steps, repeats, target, preprocessing]


# split a dataset into train/test sets
def MS_split_dataset(dataframe, config, columns):
    multi_steps, repeats, _, _ = config
    test_slice = multi_steps * repeats
    # split into standard weeks
    train, test = dataframe[:-test_slice], dataframe[-test_slice:]
    # restructure into windows of weekly data
    train = train.loc[:, columns]
    test = test.loc[:, columns]
    return train, test


# evaluate one or more weekly forecasts against expected values
def MS_evaluate_forecasts(df, multi_steps, repeats, target):
    # calculate an RMSE score for each day
    df = df.loc[:, [target, 'predicted']]
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


# summarize scores
def MS_summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def MS_to_supervised(train, n_inputs, multi_steps, target):
    X, y = list(), list()
    rows = len(train)
    columns = train.shape[1]
    # step over the entire history one time step at a time
    if columns == 1:
        train = train.values
        for i in range(rows):
            # define the end of the input sequence
            in_end = i + n_inputs
            out_end = in_end + multi_steps
            # ensure we have enough data for this instance
            if out_end <= rows:
                x_input = train[i:in_end, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y_input = train[in_end:out_end, 0]
                y.append(y_input)
    elif columns > 1:
        train_X = train.values
        train_y = train.loc[:, target].values
        for i in range(rows):
            # define the end of the input sequence
            in_end = i + n_inputs
            out_end = in_end + multi_steps
            # ensure we have enough data for this instance
            if out_end <= rows:
                X.append(train_X[i:in_end])
                y.append(train_y[in_end:out_end])
    return array(X), array(y)


# train the model
def MS_build_univar_model(train, model_configs, multi_steps, target):
    # prepare data
    n_inputs, n_filters, n_kernels, n_epochs, n_batch = model_configs
    train_x, train_y = MS_to_supervised(train, n_inputs, multi_steps, target)
    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(Conv1D(16, 3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # fit network
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model






# train the model
def MS_build_multivar_modelB(train, model_configs, multi_steps, target):
    # prepare data
    n_inputs, n_filters, n_kernels, n_epochs, n_batch = model_configs
    train_x, train_y = MS_to_supervised(train, n_inputs, multi_steps, target)
    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(Conv1D(64, n_kernels, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, n_kernels, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs))
    learning_rate = 0.001
    decay_rate = learning_rate / n_epochs
    momentum = 0.8
    clr = CyclicLR(mode='triangular2')
    #sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(loss='mse', optimizer='Adamax', metrics=['mae'])
    # fit network
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0, callbacks=[clr])
    return model



# make a forecast
def MS_forecast(model, history, test, n_inputs):
    df = test.copy()
    columns = history.shape[1]
    input_x = list()
    if columns == 1:
        history = history.iloc[-n_inputs:, 0].values
        input_x = history
        # reshape into [1, n_input, 1]
        input_x = input_x.reshape((1, len(input_x), 1))
    elif columns > 1:
        history = history.iloc[-n_inputs:].values
        input_x = history.reshape((1, history.shape[0], history.shape[1]))

    # forecast the next week
    values = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    df.loc[:, 'predicted'] = values[0]
    return df


# evaluate a single model
def MS_evaluate_model(model_builder, train, test, setting, model_configs):
    multi_steps, repeats, target, preprocessing = setting
    n_inputs, _, _, _, _ = model_configs

    # preprocess data
    scaler = None
    TargetScaler = None
    if preprocessing == 'Norm':
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(train)
        TargetScaler = MinMaxScaler(feature_range=(0, 1)).fit(train[[target]])
        train = pd.DataFrame(scaler.transform(train), columns=train.columns, index=train.index)
        test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
    elif preprocessing == 'Std':
        scaler = StandardScaler().fit(train)
        TargetScaler = StandardScaler().fit(train[[target]])
        train = pd.DataFrame(scaler.transform(train), columns=train.columns, index=train.index)
        test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)

    # fit model
    model = model_builder(train, model_configs, multi_steps, target)
    # history is a list of weekly data
    history = train.copy()
    # create predicted column
    #test = test.to_frame(name='actual')
    test.loc[:, 'predicted'] = 0
    for i in range(repeats):
        start = i * multi_steps
        end = (i + 1) * multi_steps
        # predict the week
        test.iloc[start:end] = MS_forecast(model=model, history=history, test=test.iloc[start:end], n_inputs=n_inputs)
        # get real observation and add to history for predicting the next week
        add_to_history = test.iloc[start:end, :].drop(['predicted'], axis=1)
        history = history.append(add_to_history)

    # reverse preprocess data
    if preprocessing != None:
        train = pd.DataFrame(scaler.inverse_transform(train), columns=train.columns, index=train.index)
        preds = test.loc[:, ['predicted']]
        test = pd.DataFrame(scaler.inverse_transform(test.drop(['predicted'], axis=1)), columns=test.drop(['predicted'], axis=1).columns, index=test.index)
        test['predicted'] = TargetScaler.inverse_transform(preds)

    # evaluate predictions
    score, scores = MS_evaluate_forecasts(df=test, multi_steps=multi_steps, repeats=repeats, target=target)
    return score, scores


# create a list of configs to try
def MS_cnn_model_configs(n_input, n_filters, n_kernels, n_epochs, n_batch):
    # create configs
    configs = list()
    for a in n_input:
        for b in n_filters:
            for c in n_kernels:
                for d in n_epochs:
                    for e in n_batch:
                        cfg = [a, b, c, d, e]
                        configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs


def MS_grid_search(model_builder, model_configs, train, test, setting, iterations=1):
    start = timer()
    print('total_iterations: {}'.format(len(model_configs)))
    collection = list()



    # loop through each model config
    for i in range(len(model_configs)):
        results = list()
        for x in range(1, (iterations + 1)):
            score, scores = MS_evaluate_model(model_builder=model_builder, model_configs=model_configs[i], train=train, test=test, setting=setting)
            results.append(score)
            print('config nr:', i, 'iteration:', x)



        if iterations > 1:
            avg_result = np.mean(results)
            std_result = np.std(results)
        else:
            avg_result = results[0]
            std_result = 0
        print(i)
        print(str(model_configs[i]))
        print('Result:', avg_result, 'STD:', std_result)

        collection.append([model_configs[i][0], model_configs[i][1], model_configs[i][2], model_configs[i][3],
                      model_configs[i][4], avg_result, std_result])
        print('Time: {}'.format(timer() - start))

    df = pd.DataFrame(collection, columns=['n_input', 'n_filters', 'n_kernels', 'n_epochs', 'n_batch', 'result', 'std'])
    return df



## Univariate CNN Model
#setting = CDM.MS_create_setting(multi_steps=37, repeats=5, target='Sales')
#model_configs = CDM.MS_cnn_model_configs(n_input=[180], n_filters=[64], n_kernels=[3], n_epochs=[20], n_batch=[1])
#train, test = CDM.MS_split_dataset(subset_data, config=setting, columns=['Sales'])
## evaluate model and get scores
###score, scores = DM.MS_evaluate_model(train=train, test=test, setting=setting, model_configs=model_configs)
###score, scores = DM.MS_grid_search(model_configs=model_configs, test=test, train=train, setting=setting)
#df_results = CDM.MS_grid_search(model_builder=CDM.MS_build_univar_model, model_configs=model_configs, test=test, train=train, setting=setting)



## Multivariate CNN Model
#setting = CDM.MS_create_setting(multi_steps=37, repeats=5, target='Sales', preprocessing=None)
#model_configs = CDM.MS_cnn_model_configs(n_input=[24], n_filters=[64], n_kernels=[7], n_epochs=[100], n_batch=[16])
#train, test = CDM.MS_split_dataset(subset_data, config=setting, columns=['Sales', 'Customers', 'Open', 'Month', 'Weekofyear', 'Promo', 'SchoolHoliday'])
## evaluate model and get scores
##score, scores = CDM.MS_evaluate_model(model_builder=CDM.MS_build_multivar_modelB, train=train, test=test, setting=setting, model_configs=model_configs[0])
#df_results = CDM.MS_grid_search(model_builder=CDM.MS_build_multivar_modelB, model_configs=model_configs, test=test, train=train, setting=setting, iterations=5)
##model = CDM.MS_build_multivar_modelB(train=train, model_configs=model_configs[0], multi_steps=37, target='Sales')
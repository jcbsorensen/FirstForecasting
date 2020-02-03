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
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import ConvLSTM2D
from timeit import default_timer as timer
from clr_callback import *
from keras.applications.resnet50 import ResNet50
from keras.models import Model


def MS_create_setting(multi_steps, repeats, target, category, predict_transform, minmax=None, stdiz=None, onehot_encode=None):
    return [multi_steps, repeats, target, category, predict_transform, minmax, stdiz, onehot_encode]


# split a dataset into train/test sets
def MS_split_dataset(dataframe, config, columns):
    multi_steps, repeats, _, _, _, _, _, _ = config
    test_slice = multi_steps * repeats
    # split into standard weeks
    train, test = dataframe[:-test_slice], dataframe[-test_slice:]
    # restructure into windows of weekly data
    train = train.loc[:, columns]
    test = test.loc[:, columns]
    return train, test


# evaluate one or more weekly forecasts against expected values
def MS_evaluate_forecasts(df, multi_steps, repeats, target, category):
    # calculate an RMSE score for each day
    if category != None:
        categories_count = len(df[category].unique().tolist())
    else:
        categories_count = 1
    df = df.loc[:, [target, 'predicted']]
    df.loc[:, 'rmse'] = 0
    df.loc[:, 'squared error'] = 0
    for i in range(multi_steps):
        df.iloc[i::multi_steps, 2] = np.sqrt(mean_squared_error(df.iloc[i::multi_steps, 0], df.iloc[i::multi_steps, 1]))
        df.iloc[i::multi_steps, 3] = (df.iloc[i::multi_steps, 0] - df.iloc[i::multi_steps, 1]) ** 2
    # calculate overall RMSE
    errorSum = df['squared error'].sum()
    score = np.sqrt(errorSum / (repeats * multi_steps * categories_count))
    scores = df.iloc[0:multi_steps, 2]
    return round(score, 2), scores


# convert history into inputs and outputs
def MS_to_supervised(train, n_inputs, multi_steps, target, category):
    X, y = list(), list()

    columns = train.shape[1]
    # step over the entire history one time step at a time
    if columns == 1:
        if category != None:
            categories = train[category].unique().tolist()
            for cat in categories:
                train_cat = train.loc[train[category]==cat]
                rows = len(train_cat)
                train = train_cat.values
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
        else:
            rows = len(train)
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
        if category != None:
            categories = train[category].unique().tolist()
            for cat in categories:
                train_cat = train.loc[train[category]==cat]
                rows = len(train_cat)
                train_X = train_cat.values
                train_y = train_cat.loc[:, target].values
                for i in range(rows):
                    # define the end of the input sequence
                    in_end = i + n_inputs
                    out_end = in_end + multi_steps
                    # ensure we have enough data for this instance
                    if out_end <= rows:
                        X.append(train_X[i:in_end])
                        y.append(train_y[in_end:out_end])
        else:
            rows = len(train)
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


def MS_build_univar_cnnlstm_model(train, model_configs, multi_steps, target):
    # prepare data
    n_inputs, n_nodes, n_epochs, n_batch = model_configs
    train_x, train_y = MS_to_supervised(train, n_inputs, multi_steps, target)
    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(Conv1D(n_nodes, 3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


def MS_build_multivar_cnnlstm_model(train, model_configs, multi_steps, target):
    # prepare data
    n_inputs, n_nodes, n_epochs, n_batch, n_seq, model_type = model_configs
    train_x, train_y = MS_to_supervised(train, n_inputs, multi_steps, target)
    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], n_seq, int((n_timesteps / n_seq)), n_features))
    # define the input cnn model
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(64, 3, activation='relu'), input_shape=(None, int((n_timesteps / n_seq)), n_features)))
    model.add(TimeDistributed(Conv1D(64, 3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(multi_steps))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # fit network
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


def MS_build_multivar_cnnlstm_modelB(train, model_configs, multi_steps, target, category):
    # prepare data
    n_inputs, n_nodes, n_epochs, n_batch, n_seq, model_type = model_configs
    train_x, train_y = MS_to_supervised(train, n_inputs, multi_steps, target, category)
    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], n_seq, int((n_timesteps / n_seq)), n_features))
    # define the input cnn model
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(64, 7, activation='relu'), input_shape=(None, int((n_timesteps / n_seq)), n_features)))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Conv1D(128, 4, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Conv1D(32, 1, activation='relu')))
    model.add(TimeDistributed(Conv1D(64, 1, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D((1))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(multi_steps))

    clr = CyclicLR(mode='triangular2')
    model.compile(loss='mse', optimizer='adamax', metrics=['mae'])
    # fit network
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0, callbacks=[clr])
    return model


def MS_build_multivar_convlstm_model(train, model_configs, multi_steps, target):
    # prepare data
    n_inputs, n_nodes, n_epochs, n_batch, n_seq, model_type = model_configs
    train_x, train_y = MS_to_supervised(train, n_inputs, multi_steps, target)
    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], n_seq, 1, int((n_timesteps / n_seq)), n_features))
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define the input cnn model
    model = Sequential()
    model.add(
        ConvLSTM2D(128, (1, 3), activation='sigmoid', return_sequences=True,
                   input_shape=(n_seq, 1, int((n_timesteps / n_seq)), n_features)))
    model.add(Dropout(0.2))
    model.add(ConvLSTM2D(256, (1, 3), activation='sigmoid'))
    model.add(Flatten())
    model.add(RepeatVector(7))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    clr = CyclicLR(mode='triangular2')
    custom_sgd = SGD(lr=0.000011)
    model.compile(loss='mse', optimizer=custom_sgd, metrics=['mae'])
    # fit network
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0, callbacks=[clr])
    return model


# train the model
def MS_build_resnet_model(train, model_configs, multi_steps, target, category):
    # prepare data
    n_inputs, n_nodes, n_epochs, n_batch, n_seq, model_type = model_configs
    train_x, train_y = MS_to_supervised(train, n_inputs, multi_steps, target, category)
    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], n_seq, int((n_timesteps / n_seq)), n_features))
    # define model
    resnet = ResNet50(weights=None, input_shape=(2, int((n_timesteps / n_seq)), n_features), include_top=False)
    resnet_output = resnet.output
    forecast = Dense(n_outputs)(resnet_output)
    model = Model(inputs=resnet.input, outputs=forecast)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # fit network
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# make a forecast
def MS_forecast(model, history, test, model_configs):
    n_inputs, _, _, _, n_seq, model_type = model_configs
    df = test.copy()
    columns = history.shape[1]
    input_x = list()
    if model_type == 'cnnlstm':
        history = history.iloc[-n_inputs:].values
        input_x = history.reshape((1, history.shape[0], history.shape[1]))
        input_x = input_x.reshape((input_x.shape[0], n_seq, int((n_inputs / n_seq)), columns))
    elif model_type == 'convlstm':
        history = history.iloc[-n_inputs:].values
        input_x = history.reshape((1, history.shape[0], history.shape[1]))
        input_x = input_x.reshape((input_x.shape[0], n_seq, 1, int((n_inputs / n_seq)), columns))

    # forecast the next week
    values = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    df.loc[:, 'predicted'] = values[0]
    return df


def MS_preprocess(train, test, target, predict_transform, minmax=None, stdiz=None, onehot_encode=None, oldScalers=None,
                  oldTargetScalers=None):
    # check if we need to reverse preprocessed data
    if oldScalers != None:
        # reverse preprocess data
        for scaler, targetScaler, columns, pred in zip(oldScalers, oldTargetScalers, [minmax, stdiz, onehot_encode],
                                                       ['minmax', 'stdiz', 'onehot']):
            if scaler != None:
                train[columns] = pd.DataFrame(scaler.inverse_transform(train[columns]), index=train.index)
                test[columns] = pd.DataFrame(scaler.inverse_transform(test[columns]), index=test.index)
                if predict_transform == pred:
                    test['predicted'] = targetScaler.inverse_transform(test[['predicted']])

        return train, test

    # transform data
    else:
        newScalers = list()
        newTargetScalers = list()

        scalerTypes = [minmax, stdiz, onehot_encode]
        scalerFunctions = [MinMaxScaler, StandardScaler, StandardScaler]

        for columns, scalerFunction in zip(scalerTypes, scalerFunctions):
            # MinMax Transform
            if columns != None:
                scaler = scalerFunction().fit(train[columns])
                TargetScaler = scalerFunction().fit(train[[target]])
                train[columns] = pd.DataFrame(scaler.transform(train[columns]), index=train.index)
                test[columns] = pd.DataFrame(scaler.transform(test[columns]), index=test.index)
                newScalers.append(scaler)
                newTargetScalers.append(TargetScaler)
            else:
                newScalers.append(None)
                newTargetScalers.append(None)

        return train, test, newScalers, newTargetScalers


# evaluate a single model
def MS_evaluate_model(model_builder, train, test, setting, model_configs):
    multi_steps, repeats, target, category, predict_transform, minmax, stdiz, onehot_encode = setting
    n_inputs, _, _, _, _, _ = model_configs

    # preprocess data
    train, test, scalers, targetScalers = MS_preprocess(train=train, test=test, target=target,
                                                        predict_transform=predict_transform, minmax=minmax,
                                                        stdiz=stdiz, onehot_encode=onehot_encode)

    # fit model
    model = model_builder(train, model_configs, multi_steps, target, category)
    # history is a list of weekly data

    # create predicted column
    test.loc[:, 'predicted'] = 0
    if category != None:
        categories = train[category].unique().tolist()
        for cat in categories:
            history = train.loc[train[category] == cat]
            test_cat = test.loc[test[category] == cat]
            for i in range(repeats):
                start = i * multi_steps
                end = (i + 1) * multi_steps
                # predict the week
                test_cat.iloc[start:end] = MS_forecast(model=model, history=history, test=test_cat.iloc[start:end],
                                                   model_configs=model_configs)
                # get real observation and add to history for predicting the next week
                add_to_history = test_cat.iloc[start:end, :].drop(['predicted'], axis=1)
                history = history.append(add_to_history)
            test.loc[test[category] == cat] = test_cat

    else:
        history = train.copy()
        for i in range(repeats):
            start = i * multi_steps
            end = (i + 1) * multi_steps
            # predict the week
            test.iloc[start:end] = MS_forecast(model=model, history=history, test=test.iloc[start:end],
                                               model_configs=model_configs)
            # get real observation and add to history for predicting the next week
            add_to_history = test.iloc[start:end, :].drop(['predicted'], axis=1)
            history = history.append(add_to_history)

    # reverse preprocess data
    train, test = MS_preprocess(train=train, test=test, target=target, predict_transform=predict_transform,
                                minmax=minmax, stdiz=stdiz, onehot_encode=onehot_encode,
                                oldScalers=scalers, oldTargetScalers=targetScalers)

    # evaluate predictions
    score, scores = MS_evaluate_forecasts(df=test, multi_steps=multi_steps, repeats=repeats, target=target, category=category)
    return score, scores


# create a list of configs to try
def MS_lstm_model_configs(n_input, n_nodes, n_epochs, n_batch, n_seq, model_type='cnnlstm'):
    # create configs
    configs = list()
    for a in n_input:
        for b in n_nodes:
            for c in n_epochs:
                for d in n_batch:
                    for e in n_seq:
                        cfg = [a, b, c, d, e, model_type]
                        configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs


def MS_grid_search(model_builder, model_configs, train, test, setting, iterations=1):
    start = timer()
    print('total_iterations: {}'.format(len(model_configs)))
    collection = list()

    for i in range(len(model_configs)):
        results = list()
        for x in range(1, (iterations + 1)):
            score, scores = MS_evaluate_model(model_builder=model_builder, model_configs=model_configs[i], train=train,
                                              test=test, setting=setting)
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

        collection.append(
            [model_configs[i][0], model_configs[i][1], model_configs[i][2], model_configs[i][3], avg_result,
             std_result])
        print('Time: {}'.format(timer() - start))

    df = pd.DataFrame(collection, columns=['n_input', 'n_nodes', 'n_epochs', 'n_batch', 'result', 'std'])
    return df

## Multivariate LSTM Model
# setting = ADM.MS_create_setting(multi_steps=37, repeats=5, target='Sales', preprocessing='Norm')
# model_configs = ADM.MS_lstm_model_configs(n_input=[24], n_nodes=[64], n_epochs=[100], n_batch=[16], n_seq=[2], model_type='cnnlstm')
# train, test = ADM.MS_split_dataset(subset_data, config=setting, columns=['Sales', 'Customers', 'Open', 'Month', 'Weekofyear', 'Promo', 'SchoolHoliday'])
## evaluate model and get scores
##model = ADM.MS_build_multivar_cnnlstm_model(train=train, model_configs=model_configs[0], multi_steps=7, target='Sales')
# df_result = ADM.MS_grid_search(ADM.MS_build_multivar_cnnlstm_modelB, model_configs=model_configs, train=train, test=test, setting=setting, iterations=5)


# train = subset_data.iloc[:-37, :]
# train = train.loc[:, ['Sales', 'Customers', 'Open', 'Month']]
# test = subset_data.iloc[-37:,0]
# test = test.to_frame(name='actual')
# test.loc[:, 'predicted'] = 0
# model = CDM.MS_build_multivar_model(train=train, model_configs=[300, 64, 3, 20, 16], multi_steps=37, target='Sales')
# test = CDM.MS_forecast(model=model, history=train, test=test, n_inputs=300)

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import NaiveModels as NM
from timeit import default_timer as timer
from keras.optimizers import SGD
import keras.optimizers as opt
import ARIMAModels as AM
import CNNDeepModels as CDM
import LSTMDeepModels as LDM
import TimeSeriesPlots as TSP
import AdvDeepModels as ADM
import ModelDiagnostics
from sklearn.metrics import mean_squared_error
from scipy import stats

# if __name__ == '__main__':
# Load Data
raw_data = pd.read_csv('./data/train_v2.csv')
del raw_data['Unnamed: 0']
subset_data = raw_data.loc[
    raw_data.Store.isin([4]), ['Date', 'Sales', 'Customers', 'Open', 'Month', 'Weekofyear', 'Promo',
                                     'SchoolHoliday', 'Store']]
# Map Day of Week and Set Date to Index
subset_data['Date'] = pd.to_datetime(subset_data['Date'])
subset_data['DoW'] = subset_data['Date'].dt.dayofweek
subset_data.set_index('Date', inplace=True)
# Sort Index to Ascending Dates (low to high)
subset_data.sort_index(inplace=True)
# TSP.plot_years(dataframe=subset_data, feature='Sales')
# TSP.plot_months(dataframe=subset_data, feature='Sales')


start = timer()

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from clr_callback import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import Callback


def lr_finder_model(config, X, y):
    n_inputs, n_nodes, n_epochs, n_batch, n_seq, model_type = config
    n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]
    X = X.reshape((X.shape[0], n_seq, int((n_timesteps / n_seq)), n_features))
    # y = y.reshape((y.shape[0], y.shape[1], 1))

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
    model.add(Dense(n_outputs))

    # custom_sgd = SGD(lr=1.3e-4)
    clr = CyclicLR(mode='triangular2')
    model.compile(loss='mse', optimizer='sgd', metrics=['mae'])
    lr_finder = ModelDiagnostics.LRFinder(min_lr=1e-3, max_lr=1)
    # fit network
    model.fit(X, y, epochs=5, batch_size=7, verbose=2, callbacks=[lr_finder])
    return model


columns = ['Sales', 'Customers', 'Open', 'Month', 'Weekofyear', 'Promo', 'SchoolHoliday']
#columns = ['Sales', 'Customers', 'Open', 'Month', 'Weekofyear', 'Promo', 'SchoolHoliday', 'Store']

# Univariate LSTM Model
setting = ADM.MS_create_setting(multi_steps=37, repeats=5, target='Sales', category=None, predict_transform='minmax',
                                minmax=['Sales', 'Customers'])
model_configs = ADM.MS_lstm_model_configs(n_input=[74], n_nodes=[64], n_epochs=[35], n_batch=[7], n_seq=[2],
                                          model_type='cnnlstm')
train, test = ADM.MS_split_dataset(subset_data, config=setting, columns=columns)



#train, test = pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)

#for i in [4, 5, 6]:
#    subset_store = subset_data.loc[subset_data.Store == i].sort_index()
#    sub_train, sub_test = ADM.MS_split_dataset(subset_store, config=setting, columns=columns)
#    train = train.append(sub_train)
#    test = test.append(sub_test)


# evaluate model and get scores
# score, scores = LDM.MS_evaluate_model(model_builder=LDM.MS_build_univar_cnnlstm_model, train=train, test=test, setting=setting, model_configs=[90, 64, 20, 16])


# scaler = MinMaxScaler(feature_range=(0, 1)).fit(train)
# targetScaler = MinMaxScaler(feature_range=(0, 1)).fit(train[['Sales']])
# train = pd.DataFrame(scaler.transform(train), columns=train.columns, index=train.index)
# test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
# train_x, train_y = ADM.MS_to_supervised(train=train, n_inputs=22, multi_steps=37, target='Sales')
# model = lr_finder_model(config=model_configs[0], X=train_x, y=train_y)


def MS_build_multivar_cnnlstm_modelB(config, X, y, val_X, val_y):
    # define the input cnn model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(64, 7, activation='relu'), input_shape=(None, int((22 / 2)), 8)))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Conv1D(128, 4, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Conv1D(32, 1, activation='relu')))
    model.add(TimeDistributed(Conv1D(64, 1, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D((1))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100, activation='sigmoid'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(37))

    custom_adamax = opt.adamax(lr=4e-1)
    custom_sgd = SGD(lr=4e-1, momentum=0.9)
    clr = CyclicLR(mode='triangular2')
    model.compile(loss='mse', optimizer='adamax', metrics=['mae'])
    # fit network
    model.fit(X, y, epochs=35, batch_size=7, verbose=2, validation_data=(val_X, val_y), callbacks=[clr])
    return model


#variables = ['adamax']
#df_loss, df_metric = ModelDiagnostics.multicat_grid_search_diagnostics(model_function=MS_build_multivar_cnnlstm_modelB,
#                                                                       variables=variables,
#                                                                       train=train,
#                                                                       test=test,
#                                                                       setting=setting,
#                                                                       model_configs=model_configs[0],
#                                                                       data_handler=ADM.MS_to_supervised,
#                                                                       iterations=5)
#print(df_loss.describe())
#print(df_metric.describe())
# 2288


print(timer() - start)

## plot predictions
# scaler = MinMaxScaler(feature_range=(0, 1)).fit(train)
# targetScaler = MinMaxScaler(feature_range=(0, 1)).fit(train[['Sales']])
# train = pd.DataFrame(scaler.transform(train), columns=train.columns, index=train.index)
# test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
#
df_result = ADM.MS_grid_search(ADM.MS_build_resnet_model, model_configs=model_configs, train=train, test=test, setting=setting, iterations=5)

#multi_steps, repeats, target, category, predict_transform, minmax, stdiz, onehot_encode = setting
#train, test, scalers, targetScalers = ADM.MS_preprocess(train=train, test=test, target=target,
#                                                        predict_transform=predict_transform, minmax=minmax,
#                                                        stdiz=stdiz, onehot_encode=onehot_encode)
#
#model = ADM.MS_build_multivar_cnnlstm_modelB(train=train, model_configs=model_configs[0], multi_steps=37, target='Sales', category='Store')
#testB = ADM.MS_forecast(model=model, history=train.loc[train.Store == 4], test=test.loc[test.Store == 4], model_configs=model_configs[0])
#
#train, test = ADM.MS_preprocess(train=train.loc[train.Store == 4], test=testB, target=target, predict_transform=predict_transform,
#                                minmax=minmax, stdiz=stdiz, onehot_encode=onehot_encode,
#                                oldScalers=scalers, oldTargetScalers=targetScalers)



#subset_dataB = subset_data.loc[subset_data.Store == 4]
##
#ax = plt.gca()
#subset_dataB.iloc[850:-37].plot(y='Sales', kind='line', ax=ax)
#testB.plot(y='predicted', kind='line', ax=ax)
#testB.plot(y='Sales', kind='line', label='actual', ax=ax)
#plt.show()


#[74, 64, 35, 7, 2, 'cnnlstm']
#Result: 2217.418 STD: 66.96683071491442

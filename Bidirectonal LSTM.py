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
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Bidirectional
from clr_callback import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import Callback

#%% Load Data
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
#%% Settings and Configs

columns = ['Sales', 'Customers', 'Open', 'Month', 'Weekofyear', 'Promo', 'SchoolHoliday']
#columns = ['Sales', 'Customers', 'Open', 'Month', 'Weekofyear', 'Promo', 'SchoolHoliday', 'Store']

# Univariate LSTM Model
setting = ADM.MS_create_setting(multi_steps=37, repeats=5, target='Sales', category=None, predict_transform='minmax',
                                minmax=['Sales', 'Customers'])
model_configs = ADM.MS_lstm_model_configs(n_input=[74], n_nodes=[50], n_epochs=[50], n_batch=[7], n_seq=[2],
                                          model_type='cnnlstm')
train, test = ADM.MS_split_dataset(subset_data, config=setting, columns=columns)



#%% Model Diagnostics

def MS_multivar_bidirectional_model(model_configs, X, y, val_X, val_y):
    n_inputs, n_nodes, n_epochs, n_batch, n_seq, model_type = model_configs
    n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]

    # define the input cnn model
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='sigmoid', return_sequences=True), input_shape=(n_timesteps, n_features)))
    model.add(LSTM(n_nodes))
    model.add(Dense(37))

    custom_sgd = SGD(lr=6e-3, momentum=0.9)
    clr = CyclicLR(mode='triangular2', base_lr=1e-5, max_lr=6e-3)
    model.compile(loss='mse', optimizer=custom_sgd, metrics=['mae'])
    # fit network
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=2, validation_data=(val_X, val_y), callbacks=[clr])
    return model



df_loss, df_metric = ModelDiagnostics.multicat_grid_search_diagnostics(model_function=MS_multivar_bidirectional_model,
                                                                       train=train,
                                                                       test=test,
                                                                       setting=setting,
                                                                       model_configs=model_configs,
                                                                       data_handler=LDM.MS_to_supervised,
                                                                       iterations=3)





# [37, 50, 15, 7, 2, 'cnnlstm']
# 3583



def MS_LR_model(model_configs, X, y):
    n_inputs, n_nodes, n_epochs, n_batch, n_seq, model_type = model_configs
    n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]

    # define the input cnn model
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='sigmoid', return_sequences=True), input_shape=(n_timesteps, n_features)))
    model.add(LSTM(n_nodes))
    model.add(Dense(37))

    custom_sgd = SGD(lr=4e-1, momentum=0.9)
    clr = CyclicLR(mode='triangular2')
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    lr_finder = ModelDiagnostics.LRFinder(min_lr=1e-8, max_lr=1)
    # fit network
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=2, callbacks=[lr_finder])
    return model


#ModelDiagnostics.learning_rate_finder(model_function=MS_LR_model,
#                                                                       train=train,
#                                                                       test=test,
#                                                                       setting=setting,
#                                                                       model_configs=model_configs[0],
#                                                                       data_handler=LDM.MS_to_supervised)
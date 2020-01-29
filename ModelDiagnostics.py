# univariate multi-step cnn for the power usage dataset
from numpy import array
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import keras.backend as K
import AdvDeepModels as ADM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import Dropout
from keras.layers import ConvLSTM2D
from timeit import default_timer as timer


def grid_search_diagnostics(model_function, variables, train, test, setting, model_configs, data_handler, iterations=1):
    multi_steps, repeats, target, preprocessing = setting
    n_inputs, n_nodes, n_epochs, n_batch, n_seq, model_type = model_configs

    sacler = None
    targetScaler = None
    if preprocessing == 'Norm':
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(train)
        lort = train[[target]].values
        targetScaler = MinMaxScaler(feature_range=(0, 1)).fit(lort)
        train = pd.DataFrame(scaler.transform(train), columns=train.columns, index=train.index)
        test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
    elif preprocessing == 'Std':
        scaler = StandardScaler().fit(train)
        targetScaler = StandardScaler().fit(train[[target]])
        train = pd.DataFrame(scaler.transform(train), columns=train.columns, index=train.index)
        test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)

    print('Total configs: {}'.format(len(variables)+1))
    all_loss = pd.DataFrame()
    all_metric = pd.DataFrame()

    for var in variables:
        # prepare data
        train_x, train_y = data_handler(train=train, n_inputs=n_inputs, multi_steps=multi_steps, target=target)
        val_x, val_y = data_handler(train=test, n_inputs=n_inputs, multi_steps=multi_steps, target=target)

        # cnnLSTM reshape
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        train_x = train_x.reshape((train_x.shape[0], n_seq, int((n_timesteps / n_seq)), n_features))
        val_x = val_x.reshape((val_x.shape[0], n_seq, int((n_timesteps / n_seq)), n_features))

        #convLSTM reshape
        #train_x = train_x.reshape((train_x.shape[0], n_seq, 1, int((n_timesteps / n_seq)), n_features))
        #train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        #val_x = val_x.reshape((val_x.shape[0], n_seq, 1, int((n_timesteps / n_seq)), n_features))
        #val_y = val_y.reshape((val_y.shape[0], val_y.shape[1], 1))

        train_loss_scores = pd.DataFrame()
        val_loss_scores = pd.DataFrame()
        train_metric_scores = pd.DataFrame()
        val_metric_scores = pd.DataFrame()
        loss_scores = list()
        metric_scores = list()
        for i in range(iterations):
            print('Config:', var, 'Iteration Nr:', i)
            model = model_function(var, X=train_x, y=train_y, val_X=val_x, val_y=val_y)
            if preprocessing != None:
                train_loss_scores[str(i)] = list(targetScaler.inverse_transform(np.reshape(model.history.history['loss'], (-1,1))))
                val_loss_scores[str(i)] = list(targetScaler.inverse_transform(np.reshape(model.history.history['val_loss'], (-1,1))))
                train_metric_scores[str(i)] = list(targetScaler.inverse_transform(np.reshape(model.history.history['mean_absolute_error'], (-1,1))))
                val_metric_scores[str(i)] = list(targetScaler.inverse_transform(np.reshape(model.history.history['val_mean_absolute_error'], (-1,1))))
                loss, metric = model.evaluate(val_x, val_y)
                loss_scores.append(loss)
                metric_scores.append(metric)
            else:
                train_loss_scores[str(i)] = model.history.history['loss']
                val_loss_scores[str(i)] = model.history.history['val_loss']
                train_metric_scores[str(i)] = model.history.history['mean_absolute_error']
                val_metric_scores[str(i)] = model.history.history['val_mean_absolute_error']
                loss, metric = model.evaluate(val_x, val_y)
                loss_scores.append(loss)
                metric_scores.append(metric)

        # plot train and validation loss
        plt.plot(train_loss_scores, color='blue', label='train')
        plt.plot(val_loss_scores, color='orange', label='validation')
        plt.title('model train vs validation loss [{}]'.format(var))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.show()

        # plot train and validation metric
        plt.plot(train_metric_scores, color='blue', label='train')
        plt.plot(val_metric_scores, color='orange', label='validation')
        plt.title('model train vs validation metric [{}]'.format(var))
        plt.ylabel('mae')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.show()

        all_loss[str(var)] = loss_scores
        all_metric[str(var)] = metric_scores
        if preprocessing != None:
            all_loss[str(var)] = np.sqrt(all_loss[str(var)])
            all_loss[str(var)] = targetScaler.inverse_transform(all_loss[[str(var)]])
            all_loss[str(var)] = all_loss[str(var)]**2
            all_metric[str(var)] = targetScaler.inverse_transform(all_metric[[str(var)]])

    all_loss.boxplot()
    plt.show()
    all_metric.boxplot()
    plt.show()
    return all_loss, all_metric


def multicat_grid_search_diagnostics(model_function, variables, train, test, setting, model_configs, data_handler, iterations=1):
    multi_steps, repeats, target, category, predict_transform, minmax, stdiz, onehot_encode = setting
    n_inputs, n_nodes, n_epochs, n_batch, n_seq, model_type = model_configs

    # preprocess data
    train, test, scalers, targetScalers = ADM.MS_preprocess(train=train, test=test, target=target,
                                                        predict_transform=predict_transform, minmax=minmax,
                                                        stdiz=stdiz, onehot_encode=onehot_encode)

    print('Total configs: {}'.format(len(variables)+1))
    all_loss = pd.DataFrame()
    all_metric = pd.DataFrame()

    for var in variables:
        # prepare data
        train_x, train_y = data_handler(train=train, n_inputs=n_inputs, multi_steps=multi_steps, target=target, category=category)
        val_x, val_y = data_handler(train=test, n_inputs=n_inputs, multi_steps=multi_steps, target=target, category=category)

        # cnnLSTM reshape
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        train_x = train_x.reshape((train_x.shape[0], n_seq, int((n_timesteps / n_seq)), n_features))
        val_x = val_x.reshape((val_x.shape[0], n_seq, int((n_timesteps / n_seq)), n_features))

        #convLSTM reshape
        #train_x = train_x.reshape((train_x.shape[0], n_seq, 1, int((n_timesteps / n_seq)), n_features))
        #train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        #val_x = val_x.reshape((val_x.shape[0], n_seq, 1, int((n_timesteps / n_seq)), n_features))
        #val_y = val_y.reshape((val_y.shape[0], val_y.shape[1], 1))

        train_loss_scores = pd.DataFrame()
        val_loss_scores = pd.DataFrame()
        train_metric_scores = pd.DataFrame()
        val_metric_scores = pd.DataFrame()
        loss_scores = list()
        metric_scores = list()
        for i in range(iterations):
            print('Config:', var, 'Iteration Nr:', i)
            model = model_function(var, X=train_x, y=train_y, val_X=val_x, val_y=val_y)
            loss, metric = model.evaluate(val_x, val_y)
            loss_scores.append(loss)
            metric_scores.append(metric)

            train_loss_scores[str(i)] = model.history.history['loss']
            val_loss_scores[str(i)] = model.history.history['val_loss']
            train_metric_scores[str(i)] = model.history.history['mean_absolute_error']
            val_metric_scores[str(i)] = model.history.history['val_mean_absolute_error']

            # reverse transforms (if any)
            for targetScaler in targetScalers:
                if targetScaler != None:
                    train_loss_scores[str(i)] = list(targetScaler.inverse_transform(np.reshape(model.history.history['loss'], (-1,1))))
                    val_loss_scores[str(i)] = list(targetScaler.inverse_transform(np.reshape(model.history.history['val_loss'], (-1,1))))
                    train_metric_scores[str(i)] = list(targetScaler.inverse_transform(np.reshape(model.history.history['mean_absolute_error'], (-1,1))))
                    val_metric_scores[str(i)] = list(targetScaler.inverse_transform(np.reshape(model.history.history['val_mean_absolute_error'], (-1,1))))

        # plot train and validation loss
        plt.plot(train_loss_scores, color='blue', label='train')
        plt.plot(val_loss_scores, color='orange', label='validation')
        plt.title('model train vs validation loss [{}]'.format(var))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.show()

        # plot train and validation metric
        plt.plot(train_metric_scores, color='blue', label='train')
        plt.plot(val_metric_scores, color='orange', label='validation')
        plt.title('model train vs validation metric [{}]'.format(var))
        plt.ylabel('mae')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.show()

        all_loss[str(var)] = loss_scores
        all_metric[str(var)] = metric_scores

        # reverse transforms (if any)
        for targetScaler in targetScalers:
            if targetScaler != None:
                all_loss[str(var)] = np.sqrt(all_loss[str(var)])
                all_loss[str(var)] = targetScaler.inverse_transform(all_loss[[str(var)]])
                all_loss[str(var)] = all_loss[str(var)]**2
                all_metric[str(var)] = targetScaler.inverse_transform(all_metric[[str(var)]])

    all_loss.boxplot()
    plt.show()
    all_metric.boxplot()
    plt.show()
    return all_loss, all_metric


class LRFinder(Callback):
    """
    Up-to date version: https://github.com/WittmannF/LRFinder
    Example of usage:
        from keras.models import Sequential
        from keras.layers import Flatten, Dense
        from keras.datasets import fashion_mnist
        !git clone https://github.com/WittmannF/LRFinder.git
        from LRFinder.keras_callback import LRFinder
        # 1. Input Data
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        mean, std = X_train.mean(), X_train.std()
        X_train, X_test = (X_train-mean)/std, (X_test-mean)/std
        # 2. Define and Compile Model
        model = Sequential([Flatten(),
                            Dense(512, activation='relu'),
                            Dense(10, activation='softmax')])
        model.compile(loss='sparse_categorical_crossentropy', \
                      metrics=['accuracy'], optimizer='sgd')
        # 3. Fit using Callback
        lr_finder = LRFinder(min_lr=1e-4, max_lr=1)
        model.fit(X_train, y_train, batch_size=128, callbacks=[lr_finder], epochs=2)
    """

    def __init__(self, min_lr, max_lr, mom=0.9, stop_multiplier=None,
                 reload_weights=True, batches_lr_update=5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mom = mom
        self.reload_weights = reload_weights
        self.batches_lr_update = batches_lr_update
        if stop_multiplier is None:
            self.stop_multiplier = -20 * self.mom / 3 + 10  # 4 if mom=0.9
            # 10 if mom=0
        else:
            self.stop_multiplier = stop_multiplier

    def on_train_begin(self, logs={}):
        p = self.params
        try:
            n_iterations = p['epochs'] * p['samples'] // p['batch_size']
        except:
            n_iterations = p['steps'] * p['epochs']

        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, \
                                           num=n_iterations // self.batches_lr_update + 1)
        self.losses = []
        self.iteration = 0
        self.best_loss = 0
        if self.reload_weights:
            self.model.save_weights('tmp.hdf5')

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')

        if self.iteration != 0:  # Make loss smoother using momentum
            loss = self.losses[-1] * self.mom + loss * (1 - self.mom)

        if self.iteration == 0 or loss < self.best_loss:
            self.best_loss = loss

        if self.iteration % self.batches_lr_update == 0:  # Evaluate each lr over 5 epochs

            if self.reload_weights:
                self.model.load_weights('tmp.hdf5')

            lr = self.learning_rates[self.iteration // self.batches_lr_update]
            K.set_value(self.model.optimizer.lr, lr)

            self.losses.append(loss)

        if loss > self.best_loss * self.stop_multiplier:  # Stop criteria
            self.model.stop_training = True

        self.iteration += 1

    def on_train_end(self, logs=None):
        if self.reload_weights:
            self.model.load_weights('tmp.hdf5')

        plt.figure(figsize=(12, 6))
        plt.plot(self.learning_rates[:len(self.losses)], self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.show()




#Multi-stacked LSTM model
#def multi_stacked_LSTM_model(config, X, y, val_X, val_y):
#    n_batch = config
#    # define model
#    model = Sequential()
#    model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(config, 1)))
#    model.add(LSTM(100, activation='sigmoid', return_sequences=True))
#    model.add(LSTM(100, activation='sigmoid', return_sequences=True))
#    model.add(LSTM(100, activation='sigmoid', return_sequences=True))
#    model.add(LSTM(100, activation='sigmoid', return_sequences=True))
#    model.add(LSTM(50, activation='sigmoid', return_sequences=True))
#    model.add(LSTM(50, activation='sigmoid', return_sequences=True))
#    model.add(LSTM(20, activation='sigmoid', return_sequences=True))
#    model.add(LSTM(20, activation='sigmoid'))
#    model.add(Dense(7))
#    clr = CyclicLR(mode='triangular2')
#    #custom_sgd = SGD(lr=1e-1, decay=(1e-1 / 3), momentum=0.8, nesterov=False)
#    custom_sgd = SGD(lr=0.014)
#    model.compile(loss='mse', optimizer=custom_sgd, metrics=['mae'])
#    lr_finder = ModelDiagnostics.LRFinder(min_lr=1e-4, max_lr=1)
#    # fit network
#    model.fit(X, y, epochs=5, batch_size=11, verbose=1, validation_data=(val_X, val_y), callbacks=[clr])
#    return model



# ConvLSTM multivariate multistep
#def MS_build_multivar_convlstm_model(config, X, y, val_X, val_y):
#    # define the input cnn model
#    model = Sequential()
#    model.add(ConvLSTM2D(128, (1, 3), activation='sigmoid', return_sequences=True, input_shape=(2, 1, int((12/2)), 7)))
#    model.add(Dropout(0.2))
#    model.add(ConvLSTM2D(256, (1, 3), activation='sigmoid'))
#    model.add(Flatten())
#    model.add(RepeatVector(7))
#    model.add(Dense(100, activation='relu'))
#    model.add(Dense(1))
#    clr = CyclicLR(mode='triangular2')
#    custom_sgd = SGD(lr=0.000011)
#    model.compile(loss='mse', optimizer=custom_sgd, metrics=['mae'])
#    lr_finder = ModelDiagnostics.LRFinder(min_lr=1.14e-4, max_lr=1)
#    # fit network
#    model.fit(X, y, epochs=20, batch_size=config, verbose=2, validation_data=(val_X, val_y), callbacks=[clr])
#    return model
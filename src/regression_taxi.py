# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input, MaxoutDense
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.models import Model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data_path = '../data'
time_format = '%Y-%m-%d_%X'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
adam = Adam(clipnorm=1.)


def load_dataset():
    train_dataframe = pandas.read_csv(os.path.join(data_path, 'train.dat'), header=None)
    train1_dataframe = pandas.read_csv(os.path.join(data_path, 'train1.dat'), header=None)
    train2_dataframe = pandas.read_csv(os.path.join(data_path, 'train2.dat'), header=None)
    train3_dataframe = pandas.read_csv(os.path.join(data_path, 'train3.dat'), header=None)
    test_dataframe = pandas.read_csv(os.path.join(data_path, 'test.dat'), header=None)
    valid_dataframe = pandas.read_csv(os.path.join(data_path, 'valid.dat'), header=None)
    train_dataset = train_dataframe.values.astype('float32')
    train1_dataset = train1_dataframe.values.astype('float32')
    train2_dataset = train2_dataframe.values.astype('float32')
    train3_dataset = train3_dataframe.values.astype('float32')
    test_dataset = test_dataframe.values.astype('float32')
    valid_dataset = valid_dataframe.values.astype('float32')
    return train_dataset, train1_dataset, train2_dataset, train3_dataset, test_dataset, valid_dataset


def load_samples():
    samples_dataframe = pandas.read_csv(os.path.join(data_path, 'train_sample.dat'), header=None)
    samples_dataset = samples_dataframe.values
    return samples_dataset


def maxout_model():
    model = Sequential()
    model.add(MaxoutDense(128, nb_feature=128, input_dim=22))
    model.add(MaxoutDense(64, nb_feature=64))
    model.add(MaxoutDense(32, nb_feature=32))

    model.add(Dense(1, init='zero'))

    model.compile(loss='mape', optimizer='adam')
    return model


def mlp_model():
    model = Sequential()
    model.add(Dense(128, init='glorot_normal', activation='relu', input_dim=21))
    model.add(Dense(64, init='glorot_normal', activation='relu',))
    model.add(Dense(32, init='glorot_normal', activation='relu',))
    model.add(Dense(1, init='zero'))
    model.compile(loss='mape', optimizer='adam')
    return model


def ae():
    encoding_dim = 16
    model = Sequential()
    model.add(Dense(encoding_dim, input_dim=22, activation='relu'))
    model.add(Dense(22, activation='sigmoid'))
    model.compile(optimizer=adam, loss='mse')
    return model


def make_submit_mlp():
    train, train1, train2, train3, test, valid = load_dataset()

    x_train = train[:, 0:21]
    y_train = train[:, 21]

    x_train1 = train1[:, 0:21]
    y_train1 = train1[:, 21]

    x_train2 = train2[:, 0:21]
    y_train2 = train2[:, 21]

    x_train3 = train2[:, 0:21]
    y_train3 = train2[:, 21]

    x_valid = valid[:, 0:21]
    y_valid = valid[:, 21]

    x_test = test[:, 0:21]

    x_scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train.reshape(-1, 21))
    y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(y_train.reshape(-1, 1))
    x_train1 = (x_scaler.transform(x_train1.reshape(-1, 21)))
    y_train1 = (y_scaler.transform(y_train1.reshape(-1, 1)))
    x_train2 = (x_scaler.transform(x_train2.reshape(-1, 21)))
    y_train2 = (y_scaler.transform(y_train2.reshape(-1, 1)))
    x_train3 = (x_scaler.transform(x_train3.reshape(-1, 21)))
    y_train3 = (y_scaler.transform(y_train3.reshape(-1, 1)))
    x_valid = (x_scaler.transform(x_valid.reshape(-1, 21)))
    y_valid = (y_scaler.transform(y_valid.reshape(-1, 1)))

    proposed_model1 = mlp_model()
    proposed_model1.fit(x_train1, y_train1, nb_epoch=10, batch_size=128, verbose=1, validation_data=(x_valid, y_valid))
    proposed_model2 = mlp_model()
    proposed_model2.fit(x_train2, y_train2, nb_epoch=10, batch_size=128, verbose=1, validation_data=(x_valid, y_valid))
    proposed_model3 = mlp_model()
    proposed_model3.fit(x_train3, y_train3, nb_epoch=10, batch_size=128, verbose=1, validation_data=(x_valid, y_valid))

    y_predict = list()
    for test_sample in x_test:
        if test_sample[19] > 12065.31:
            test_sample = x_scaler.transform(test_sample.reshape(-1, 21))
            sample_result = proposed_model3.predict(test_sample, batch_size=1)
            sample_predict = y_scaler.inverse_transform(sample_result.reshape(-1, 1))
            y_predict.extend(sample_predict)
        elif 12065.31 >= test_sample[19] > 7000:
            test_sample = x_scaler.transform(test_sample.reshape(-1, 21))
            sample_result = proposed_model2.predict(test_sample, batch_size=1)
            sample_predict = y_scaler.inverse_transform(sample_result.reshape(-1, 1))
            y_predict.extend(sample_predict)
        else:
            test_sample = x_scaler.transform(test_sample.reshape(-1, 21))
            sample_result = proposed_model1.predict(test_sample, batch_size=1)
            sample_predict = y_scaler.inverse_transform(sample_result.reshape(-1, 1))
            y_predict.extend(sample_predict)

    y_predict = np.asarray(y_predict)

    trip_id = np.array(range(1, len(y_predict)+1))
    results = np.column_stack((trip_id, y_predict))
    timestamp = time.strftime(time_format, time.gmtime(time.time()))
    np.savetxt('rst_' + timestamp + '.csv', results, header='pathid,time', comments='', fmt='%d,%f')


def make_submit_maxout():
    train, test, valid = load_dataset()
    x_train = train[:, 0:22]
    y_train = train[:, 22]

    x_valid = valid[:, 0:22]
    y_valid = valid[:, 22]

    x_test = test[:, 0:22]

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = (x_scaler.fit_transform(x_train.reshape(-1, 22)))
    y_train = (y_scaler.fit_transform(y_train.reshape(-1, 1)))
    x_valid = (x_scaler.fit_transform(x_valid.reshape(-1, 22)))
    y_valid = (y_scaler.fit_transform(y_valid.reshape(-1, 1)))
    x_test = (x_scaler.fit_transform(x_test.reshape(-1, 22)))

    proposed_model = maxout_model()
    proposed_model.fit(x_train, y_train, nb_epoch=5, batch_size=128, verbose=1, validation_data=(x_valid, y_valid))
    y_predict = y_scaler.inverse_transform(proposed_model.predict(x_test).reshape(-1, 1))

    trip_id = np.array(range(1, len(y_predict)+1))
    results = np.column_stack((trip_id, y_predict))
    timestamp = time.strftime(time_format, time.gmtime(time.time()))
    np.savetxt('rst_' + timestamp + '.csv', results, header='pathid,time', comments='', fmt='%d,%f')


def make_submit_ae_mlp():
    train, test, valid = load_dataset()
    x_train = train[:, 0:22]
    y_train = train[:, 22]

    x_valid = valid[:, 0:22]
    y_valid = valid[:, 22]

    x_test = test[:, 0:22]

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = (x_scaler.fit_transform(x_train.reshape(-1, 22)))
    y_train = (y_scaler.fit_transform(y_train.reshape(-1, 1)))
    x_valid = (x_scaler.fit_transform(x_valid.reshape(-1, 22)))
    y_valid = (y_scaler.fit_transform(y_valid.reshape(-1, 1)))
    x_test = (x_scaler.fit_transform(x_test.reshape(-1, 22)))

    autoencoder = ae()
    autoencoder.fit(x_train, x_train, batch_size=256, nb_epoch=5)
    encoder = autoencoder.layers[0]
    encoder.build = lambda: None
    model = Sequential()
    model.add(encoder)
    model.add(Dense(512, init='glorot_normal', activation='relu'))
    model.add(Dense(256, init='glorot_normal', activation='relu'))
    model.add(Dense(128, init='glorot_normal', activation='relu'))
    model.add(Dense(64, init='glorot_normal', activation='relu'))
    model.add(Dense(32, init='glorot_normal', activation='relu'))
    model.add(Dense(1, init='zero', activation='linear'))

    model.compile(optimizer='adam', loss='mape')
    model.fit(x_train, y_train, nb_epoch=5, batch_size=128, verbose=1, validation_data=(x_valid, y_valid))
    y_predict = y_scaler.inverse_transform(model.predict(x_test).reshape(-1, 1))

    trip_id = np.array(range(1, len(y_predict)+1))
    results = np.column_stack((trip_id, y_predict))
    timestamp = time.strftime(time_format, time.gmtime(time.time()))
    np.savetxt('rst_' + timestamp + '.csv', results, header='pathid,time', comments='', fmt='%d,%f')


if __name__ == '__main__':
    # make_submit_maxout()
    # make_submit_ae_mlp()
    make_submit_mlp()

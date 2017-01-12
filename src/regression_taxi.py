# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

data_path = '../data'
time_format = '%Y-%m-%d_%X'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def load_dataset():
    train_dataframe = pandas.read_csv(os.path.join(data_path, 'train.dat'), header=None)
    test_dataframe = pandas.read_csv(os.path.join(data_path, 'test.dat'), header=None)
    train_dataset = train_dataframe.values.astype('float32')
    test_dataset = test_dataframe.values.astype('float32')
    return train_dataset, test_dataset


def load_samples():
    samples_dataframe = pandas.read_csv(os.path.join(data_path, 'train_sample.dat'), header=None)
    samples_dataset = samples_dataframe.values
    return samples_dataset


# define base mode
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(512, input_dim=22, init='glorot_normal', activation='relu', W_regularizer=l2(0.01)))
    model.add(Dense(128, init='glorot_normal', activation='relu', W_regularizer=l2(0.01)))
    model.add(Dense(64, init='glorot_normal', activation='relu', W_regularizer=l2(0.01)))
    model.add(Dense(32, init='glorot_normal', activation='relu', W_regularizer=l2(0.01)))
    model.add(Dense(1, init='zero', activation='linear'))
    # Compile model
    model.compile(loss='mape', optimizer='adam')
    return model


def mlp_model():
    model = Sequential()
    model.add(Dense(128, input_dim=22, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Activation('linear'))
    model.add(Dense(64, activation='relu'))
    model.add(Activation('linear'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mape', optimizer='adam')
    return model


def model_wrapper():
    # evaluate model with standardized dataset
    estimators = list()
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=10, batch_size=128, verbose=1)))
    pipeline = Pipeline(estimators)
    return pipeline


def make_submit():
    train, test = load_dataset()
    x_train = train[:, 0:22]
    y_train = train[:, 22]

    x_test = test[:, 0:22]
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = (x_scaler.fit_transform(x_train.reshape(-1, 22)))
    y_train = (y_scaler.fit_transform(y_train.reshape(-1, 1)))
    proposed_model = mlp_model()
    proposed_model.fit(x_train, y_train, nb_epoch=100, batch_size=128, verbose=2)
    y_predict = y_scaler.inverse_transform(proposed_model.predict(x_test).reshape(-1, 1))

    trip_id = np.array(range(1, len(y_predict)+1))
    results = np.column_stack((trip_id, y_predict))
    timestamp = time.strftime(time_format, time.gmtime(time.time()))
    np.savetxt('rst_' + timestamp + '.csv', results, header='pathid,time', comments='', fmt='%d,%f')


def model_evaluation():
    samples = load_samples()
    x_samples = samples[:, 0:22]
    y_samples = samples[:, 22]
    proposed_model = model_wrapper()
    x_train, x_test, y_train, y_test = train_test_split(x_samples, y_samples, test_size=0.1, random_state=seed)
    proposed_model.fit(x_train, y_train)
    score = proposed_model.score(x_test, y_test)
    print '\n'
    print ('Standardized: %.4f MAPE' % score)
    print '\n'


if __name__ == '__main__':
    # model_evaluation()
    make_submit()

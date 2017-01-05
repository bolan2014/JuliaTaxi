# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data_path = '../data'
time_format = '%Y-%m-%d_%X'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def load_dataset():
    train_dataframe = pandas.read_csv(os.path.join(data_path, 'train.dat'), header=None)
    test_dataframe = pandas.read_csv(os.path.join(data_path, 'test.dat'), header=None)
    train_dataset = train_dataframe.values
    test_dataset = test_dataframe.values
    return train_dataset, test_dataset


def load_samples():
    samples_dataframe = pandas.read_csv(os.path.join(data_path, 'train_sample.dat'), header=None)
    samples_dataset = samples_dataframe.values
    return samples_dataset


# define base mode
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=22, init='normal', activation='relu'))
    model.add(Dense(128, init='normal', activation='relu'))
    model.add(Dense(64, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
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
    proposed_model = model_wrapper()
    proposed_model.fit(x_train, y_train)
    y_predict = proposed_model.predict(x_test)

    trip_id = np.array(range(1, len(y_predict)+1))
    results = np.column_stack((trip_id, y_predict))
    timestamp = time.strftime(time_format, time.gmtime(time.time()))
    np.savetxt('rst_' + timestamp + '.csv', results, header='pathid,time', comments='', fmt='%d,%f')


def model_evaluation():
    samples = load_samples()
    x_samples = samples[:, 0:22]
    y_samples = samples[:, 22]
    proposed_model = model_wrapper()
    k_fold = KFold(n_splits=10, random_state=seed)
    # results = cross_val_score(proposed_model, x_samples, y_samples, cv=k_fold)
    # print '\n'
    # print ('Standardized: %.2f (%.2f) MSE' % (results.mean(), results.std()))
    # print '\n'

    mean_absolute_percentage_errors = []
    for train, test in k_fold.split(x_samples):
        proposed_model.fit(x_samples[train], y_samples[train])
        y_predict = proposed_model.predict(x_samples[test])
        y_test = y_samples[test]
        error = []
        for i in range(len(y_predict)):
            error.append(round(abs(y_predict[i] - y_test[i]) / y_test[i], 3))
        mape = sum(error) / len(error)
        mean_absolute_percentage_errors.append(mape)
    print '\n'
    print ('Mean MAPE: %.4f' % (sum(mean_absolute_percentage_errors) / len(mean_absolute_percentage_errors)))
    print '\n'


if __name__ == '__main__':
    model_evaluation()
    # make_submit()

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
dim = 22
x_slice = range(0, 22)


def load_dataset():
    train_dataframe = pandas.read_csv(os.path.join(data_path, 'train.dat'), header=None)
    test_dataframe = pandas.read_csv(os.path.join(data_path, 'test.dat'), header=None)
    valid_dataframe = pandas.read_csv(os.path.join(data_path, 'valid.dat'), header=None)
    train_dataset = train_dataframe.values.astype('float32')
    test_dataset = test_dataframe.values.astype('float32')
    valid_dataset = valid_dataframe.values.astype('float32')
    return train_dataset, test_dataset, valid_dataset


def load_samples():
    samples_dataframe = pandas.read_csv(os.path.join(data_path, 'train_sample.dat'), header=None)
    samples_dataset = samples_dataframe.values
    return samples_dataset


def maxout_model():
    model = Sequential()
    model.add(MaxoutDense(240, nb_feature=5, input_dim=dim))
    model.add(MaxoutDense(240, nb_feature=5, input_dim=dim))
    model.add(Dense(1, init='zero'))

    model.compile(loss='mape', optimizer='adam')
    return model


def mlp_model():
    model = Sequential()
    model.add(Dense(256, init='glorot_uniform', activation='relu', W_regularizer=l2(0.001), input_dim=dim))
    model.add(Dense(128, init='glorot_uniform', activation='relu', W_regularizer=l2(0.001)))
    model.add(Dense(64, init='glorot_uniform', activation='relu', W_regularizer=l2(0.001)))
    model.add(Dense(32, init='glorot_uniform', activation='relu', W_regularizer=l2(0.001)))
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


def make_submit_all():
    train_raw, test, valid = load_dataset()

    train = list()
    for train_sample in train_raw:
        if not train_sample[22]:
            continue
        else:
            train.append(train_sample)

    train = np.asarray(train)

    x_train = train[:, x_slice]
    y_train = train[:, 22]
    x_test = test[:, x_slice]
    x_valid = valid[:, x_slice]
    y_valid = valid[:, 22]

    # x_scale = StandardScaler()
    x_scale = MinMaxScaler()
    x_train = x_scale.fit_transform(x_train)
    x_valid = x_scale.transform(x_valid)
    x_test = x_scale.transform(x_test)

    proposed_model = mlp_model()
    proposed_model.fit(x_train, y_train, nb_epoch=5, batch_size=128, verbose=1, validation_data=(x_valid, y_valid))

    y_predict = proposed_model.predict(x_test)

    trip_id = np.array(range(1, len(y_predict)+1))
    results = np.column_stack((trip_id, y_predict))
    timestamp = time.strftime(time_format, time.gmtime(time.time()))
    np.savetxt('rst_' + timestamp + '.csv', results, header='pathid,time', comments='', fmt='%d,%f')


def make_submit():
    train, test, valid = load_dataset()
    train1 = list()
    train2 = list()
    train3 = list()
    train4 = list()
    train5 = list()
    train6 = list()
    train7 = list()
    train8 = list()

    for train_sample in train:
        if not train_sample[22]:
            continue
        if train_sample[20] > 14000:
            train8.append(train_sample)
        elif 14000 >= train_sample[20] > 10000:
            train7.append(train_sample)
        elif 10000 >= train_sample[20] > 8000:
            train6.append(train_sample)
        elif 8000 >= train_sample[20] > 6000:
            train5.append(train_sample)
        elif 6000 >= train_sample[20] > 5000:
            train4.append(train_sample)
        elif 5000 >= train_sample[20] > 4000:
            train3.append(train_sample)
        elif 4000 >= train_sample[20] > 3000:
            train2.append(train_sample)
        else:
            train1.append(train_sample)

    train1 = np.asarray(train1)
    train2 = np.asarray(train2)
    train3 = np.asarray(train3)
    train4 = np.asarray(train4)
    train5 = np.asarray(train5)
    train6 = np.asarray(train6)
    train7 = np.asarray(train7)
    train8 = np.asarray(train8)

    # x_train = train[:, 0:21]
    # y_train = train[:, 21]

    x_train1 = train1[:, 0:22]
    y_train1 = train1[:, 22]

    x_train2 = train2[:, 0:22]
    y_train2 = train2[:, 22]

    x_train3 = train3[:, 0:22]
    y_train3 = train3[:, 22]

    x_train4 = train4[:, 0:22]
    y_train4 = train4[:, 22]

    x_train5 = train5[:, 0:22]
    y_train5 = train5[:, 22]

    x_train6 = train6[:, 0:22]
    y_train6 = train6[:, 22]

    x_train7 = train7[:, 0:22]
    y_train7 = train7[:, 22]

    x_train8 = train8[:, 0:22]
    y_train8 = train8[:, 22]

    x_valid = valid[:, 0:22]
    y_valid = valid[:, 22]

    x_test = test[:, 0:22]

    # x_scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train.reshape(-1, 21))
    # y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(y_train.reshape(-1))
    # x_train1 = (x_scaler.transform(x_train1.reshape(-1, 21)))
    # y_train1 = (y_scaler.transform(y_train1.reshape(-1)))
    # x_train2 = (x_scaler.transform(x_train2.reshape(-1, 21)))
    # y_train2 = (y_scaler.transform(y_train2.reshape(-1)))
    # x_train3 = (x_scaler.transform(x_train3.reshape(-1, 21)))
    # y_train3 = (y_scaler.transform(y_train3.reshape(-1)))
    # x_train4 = (x_scaler.transform(x_train4.reshape(-1, 21)))
    # y_train4 = (y_scaler.transform(y_train4.reshape(-1)))
    # x_train5 = (x_scaler.transform(x_train5.reshape(-1, 21)))
    # y_train5 = (y_scaler.transform(y_train5.reshape(-1)))
    # x_train6 = (x_scaler.transform(x_train6.reshape(-1, 21)))
    # y_train6 = (y_scaler.transform(y_train6.reshape(-1)))
    # x_valid = (x_scaler.transform(x_valid.reshape(-1, 21)))
    # y_valid = (y_scaler.transform(y_valid.reshape(-1)))

    proposed_model1 = mlp_model()
    proposed_model1.fit(x_train1, y_train1, nb_epoch=5, batch_size=128, verbose=1)
    proposed_model2 = mlp_model()
    proposed_model2.fit(x_train2, y_train2, nb_epoch=5, batch_size=128, verbose=1)
    proposed_model3 = mlp_model()
    proposed_model3.fit(x_train3, y_train3, nb_epoch=5, batch_size=128, verbose=1)
    proposed_model4 = mlp_model()
    proposed_model4.fit(x_train4, y_train4, nb_epoch=5, batch_size=128, verbose=1)
    proposed_model5 = mlp_model()
    proposed_model5.fit(x_train5, y_train5, nb_epoch=5, batch_size=128, verbose=1)
    proposed_model6 = mlp_model()
    proposed_model6.fit(x_train6, y_train6, nb_epoch=5, batch_size=128, verbose=1)
    proposed_model7 = mlp_model()
    proposed_model7.fit(x_train7, y_train7, nb_epoch=5, batch_size=128, verbose=1)
    proposed_model8 = mlp_model()
    proposed_model8.fit(x_train8, y_train8, nb_epoch=5, batch_size=128, verbose=1)

    y_predict = list()
    for test_sample in x_test:
        if test_sample[20] > 14000:
            test_sample = test_sample.reshape(-1, dim)
            sample_result = proposed_model8.predict(test_sample, batch_size=128)
            y_predict.extend(sample_result)
        elif 14000 >= test_sample[20] > 10000:
            test_sample = test_sample.reshape(-1, dim)
            sample_result = proposed_model7.predict(test_sample, batch_size=128)
            y_predict.extend(sample_result)
        elif 10000 >= test_sample[20] > 8000:
            test_sample = test_sample.reshape(-1, dim)
            sample_result = proposed_model6.predict(test_sample, batch_size=128)
            y_predict.extend(sample_result)
        elif 8000 >= test_sample[20] > 6000:
            test_sample = test_sample.reshape(-1, dim)
            sample_result = proposed_model5.predict(test_sample, batch_size=128)
            y_predict.extend(sample_result)
        elif 6000 >= test_sample[20] > 5000:
            test_sample = test_sample.reshape(-1, dim)
            sample_result = proposed_model4.predict(test_sample, batch_size=128)
            y_predict.extend(sample_result)
        elif 5000 >= test_sample[20] > 4000:
            test_sample = test_sample.reshape(-1, dim)
            sample_result = proposed_model3.predict(test_sample, batch_size=128)
            y_predict.extend(sample_result)
        elif 4000 >= test_sample[20] > 3000:
            test_sample = test_sample.reshape(-1, dim)
            sample_result = proposed_model2.predict(test_sample, batch_size=128)
            y_predict.extend(sample_result)
        else:
            test_sample = test_sample.reshape(-1, dim)
            sample_result = proposed_model1.predict(test_sample, batch_size=128)
            y_predict.extend(sample_result)

    y_predict = np.asarray(y_predict)

    y_val = list()
    for val_sample in x_valid:
        if val_sample[20] > 14000:
            val_sample = val_sample.reshape(-1, dim)
            sample_result = proposed_model6.predict(val_sample, batch_size=128)
            y_val.extend(sample_result)
        elif 14000 >= val_sample[20] > 10000:
            val_sample = val_sample.reshape(-1, dim)
            sample_result = proposed_model5.predict(val_sample, batch_size=128)
            y_val.extend(sample_result)
        elif 10000 >= val_sample[20] > 8000:
            val_sample = val_sample.reshape(-1, dim)
            sample_result = proposed_model4.predict(val_sample, batch_size=128)
            y_val.extend(sample_result)
        elif 8000 >= val_sample[20] > 6000:
            val_sample = val_sample.reshape(-1, dim)
            sample_result = proposed_model3.predict(val_sample, batch_size=128)
            y_val.extend(sample_result)
        elif 6000 >= val_sample[20] > 5000:
            val_sample = val_sample.reshape(-1, dim)
            sample_result = proposed_model2.predict(val_sample, batch_size=128)
            y_val.extend(sample_result)
        elif 5000 >= val_sample[20] > 4000:
            val_sample = val_sample.reshape(-1, dim)
            sample_result = proposed_model2.predict(val_sample, batch_size=128)
            y_val.extend(sample_result)
        elif 4000 >= val_sample[20] > 3000:
            val_sample = val_sample.reshape(-1, dim)
            sample_result = proposed_model2.predict(val_sample, batch_size=128)
            y_val.extend(sample_result)
        else:
            val_sample = val_sample.reshape(-1, dim)
            sample_result = proposed_model1.predict(val_sample, batch_size=128)
            y_val.extend(sample_result)

    y_val = np.asarray(y_val)

    error = []
    for i in range(len(y_val)):
        error.append(round(abs(y_val[i] - y_valid[i]) / y_valid[i], 3))
    mape = sum(error) / len(error)

    print '\n'
    print ('MAPE: %.4f' % mape)
    print '\n'

    trip_id = np.array(range(1, len(y_predict)+1))
    results = np.column_stack((trip_id, y_predict))
    timestamp = time.strftime(time_format, time.gmtime(time.time()))
    np.savetxt('rst_' + timestamp + '.csv', results, header='pathid,time', comments='', fmt='%d,%f')


if __name__ == '__main__':
    # make_submit()
    make_submit_all()

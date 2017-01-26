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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

data_path = '../data'
time_format = '%Y-%m-%d_%X'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


train_dataframe = pandas.read_csv(os.path.join(data_path, 'train.dat'), header=None)
test_dataframe = pandas.read_csv(os.path.join(data_path, 'test.dat'), header=None)
valid_dataframe = pandas.read_csv(os.path.join(data_path, 'valid.dat'), header=None)
train_dataset = train_dataframe.values.astype('float32')
test_dataset = test_dataframe.values.astype('float32')
valid_dataset = valid_dataframe.values.astype('float32')

train = list()
for train_sample in train_dataset:
    if not train_sample[21]:
        continue
    else:
        train.append(train_sample)

train = np.asarray(train)

x_train = train[:, 0:21]
y_train = train[:, 21]
x_test = test_dataset[:, 0:21]
x_valid = valid_dataset[:, 0:21]
y_valid = valid_dataset[:, 21]

ss_X = MinMaxScaler()
ss_y = MinMaxScaler()

x_train = ss_X.fit_transform(x_train)
y_train = ss_y.fit_transform(y_train)
x_test = ss_X.transform(x_test)
x_valid = ss_X.transform(x_valid)
y_valid = ss_y.transform(y_valid)

# Random Forest
rfr = RandomForestRegressor(n_estimators=50, random_state=seed)
rfr.fit(x_train, y_train)
rfr_y_predict = rfr.predict(x_valid)

print 'Begin to evaluate ...'
print 'The MAPE value of Random Forest is', mean_absolute_percentage_error(y_valid, rfr_y_predict)

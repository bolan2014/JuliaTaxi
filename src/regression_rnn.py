# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
data_path = '../data'
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM
import pandas
import os

maxlen = 200
batch_size = 32


print('Loading data...')
with open('../data/train.dat') as f:
    train = [line.strip().split(',') for line in f.readlines()]
    for line in train:
        for idx, x in enumerate(line):
            line[idx] = float(x)
    train = np.array(train)

# with open('../data/valid.dat') as f:
#     valid = [line.strip().split(',') for line in f.readlines()]
#     for line in valid:
#         for idx, x in enumerate(line):
#             line[idx] = float(x)
#     valid = np.array(valid)

with open('../data/test.dat') as f:
    test = [line.strip().split(',') for line in f.readlines()]
    for line in test:
        for idx, x in enumerate(line):
            line[idx] = float(x)
    test = np.array(test)

x_train = [sample[0:-1] for sample in train]
x_train = np.array(x_train)
y_train = [sample[-1] for sample in train]
y_train = np.array(y_train)

# x_valid = [sample[0:-1] for sample in valid]
# x_valid = np.array(x_valid)
# y_valid = [sample[-1] for sample in valid]
# y_valid = np.array(y_valid)

x_test = [sample[0:-1] for sample in test]
x_test = np.array(x_test)


print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(x_train, maxlen=maxlen)
X_test = sequence.pad_sequences(x_test, maxlen=maxlen)
X_train = X_train.reshape(-1, maxlen, 1)
X_test = X_test.reshape(-1, maxlen, 1)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, input_shape=(maxlen, 1)))  # try using a GRU instead, for fun
model.add(Dense(1))

# try using different optimizers and different optimizer configs
model.compile(loss='mape',
              optimizer='adam')

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15, verbose=1)
# score, acc = model.evaluate(X_test, y_test,
#                            batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)

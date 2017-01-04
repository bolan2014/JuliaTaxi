#-*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from data_input import get_train, get_test

ftrain, ftest = 'data/train.dat', 'data/test.dat'

X_train, y_train = get_train(ftrain)

print 'Data is ready ...'

#训练集比例
#scale = int(0.8 * len(X))

#X_train, X_test = X[:scale], X[scale:]
#y_train, y_test = y[:scale], y[scale:]

rf = RandomForestRegressor(max_depth=20, verbose=True)

rf.fit(X_train, y_train)

print 'Random forest training complete ...'

X_test = get_test(ftest)

y_test = rf.predict(X_test)

print 'Prediction is ready ...'

#写文件
tripId = np.array(range(1, len(y_test)+1))

y_predict = np.column_stack((tripId, y_test))

np.savetxt('rst.csv', y_predict, header='pathid,time', comments='', fmt='%d,%f')

#error = []

#for i in range(len(y_test)):
#	error.append(round(abs(y_test[i]-rst[i]) / y_test[i], 3))

#print 'error:', sum(error) / len(error)

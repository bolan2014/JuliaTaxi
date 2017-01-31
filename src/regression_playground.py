# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

data_path = '../data'
time_format = '%Y-%m-%d_%X'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def make_submit(model_name, y_pred):
    trip_id = np.array(range(1, len(y_pred) + 1))
    results = np.column_stack((trip_id, y_pred))
    timestamp = time.strftime(time_format, time.gmtime(time.time()))
    np.savetxt(model_name + '_' + timestamp + '.csv', results, header='pathid,time', comments='', fmt='%d,%f')


train_dataframe = pandas.read_csv(os.path.join(data_path, 'train.dat'), header=None)
test_dataframe = pandas.read_csv(os.path.join(data_path, 'test.dat'), header=None)
valid_dataframe = pandas.read_csv(os.path.join(data_path, 'valid.dat'), header=None)
train_dataset = train_dataframe.values.astype('float32')
test_dataset = test_dataframe.values.astype('float32')
valid_dataset = valid_dataframe.values.astype('float32')


x_train = train_dataset[:, 0:21]
y_train = train_dataset[:, 21]
x_test = test_dataset[:, 0:21]
x_valid = valid_dataset[:, 0:21]
y_valid = valid_dataset[:, 21]

# ss_X = StandardScaler()
#
# x_train = ss_X.fit_transform(x_train)
# x_test = ss_X.transform(x_test)
# x_valid = ss_X.transform(x_valid)

# # Random Forest
# rfr = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=20)
# rfr.fit(x_train, y_train)
# rfr_y_predict = ss_y.inverse_transform(rfr.predict(x_valid))
# print 'The MAPE value of Random Forest is', mean_absolute_percentage_error(y_valid, rfr_y_predict)
# y_rfr_predict = ss_y.inverse_transform(rfr.predict(x_test))
# make_submit('random_forest', y_rfr_predict)

# # ExtraTrees Regressor
# etr = ExtraTreesRegressor(random_state=seed, n_estimators=50, n_jobs=20)
# etr.fit(x_train, y_train)
# etr_y_predict = etr.predict(x_valid)
# print 'The MAPE value of Extra Tree is', mean_absolute_percentage_error(y_valid, etr_y_predict)
# y_etr_predict = etr.predict(x_test)
# make_submit('extremely_randomized_trees', y_etr_predict)

# # KNN
# knr = KNeighborsRegressor(weights='distance', n_jobs=20)
# knr.fit(x_train, y_train)
# knr_y_predict = knr.predict(x_valid)
# print 'The MAPE value of KNN is', mean_absolute_percentage_error(y_valid, knr_y_predict)
# y_knr_predict = knr.predict(x_test)
# make_submit('knn', y_knr_predict)

reg = list()
reg.append(('standardize', StandardScaler()))
reg.append(('etr', ExtraTreesRegressor()))
pipeline = Pipeline(reg)
parameters = {'etr__n_estimators': np.linspace(10, 100, num=10),
              'etr__max_features': ('auto', 'sqrt', 'log2', None),
              'etr__max_depth': [10, 20, 50, 100, None]}
gs = GridSearchCV(pipeline, parameters, verbose=2, refit=True, cv=20, n_jobs=-1)
gs.fit(x_train, y_train)
print gs.best_params_
print gs.best_score_

gs_y_predict = gs.predict(x_valid)
print 'The MAPE value of Grid Search is', mean_absolute_percentage_error(y_valid, gs_y_predict)

y_gs_predict = gs.predict(x_test)
make_submit('grid_search', y_gs_predict)

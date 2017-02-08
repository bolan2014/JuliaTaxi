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
from xgboost import XGBRegressor

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

# x_slice = [0, 1, 2, 19, 20]
x_slice = range(0, 22)

# train1 = list()
# train2 = list()
# train3 = list()
#
# for train_sample in train_dataset:
#     if not train_sample[22]:
#         continue
#     if train_sample[20] > 9000:
#         train3.append(train_sample)
#     elif 9000 >= train_sample[20] > 5000:
#         train2.append(train_sample)
#     else:
#         train1.append(train_sample)
#
# train1 = np.asarray(train1)
# train2 = np.asarray(train2)
# train3 = np.asarray(train3)
#
# x_train1 = train1[:, 0:22]
# y_train1 = train1[:, 22]
#
# x_train2 = train2[:, 0:22]
# y_train2 = train2[:, 22]
#
# x_train3 = train3[:, 0:22]
# y_train3 = train3[:, 22]
#
# print len(x_train1)
# print len(x_train2)
# print len(x_train3)

x_train = train_dataset[:, x_slice]
y_train = train_dataset[:, 22]
x_test = test_dataset[:, x_slice]
x_valid = valid_dataset[:, x_slice]
y_valid = valid_dataset[:, 22]

ss_X = StandardScaler()
# ss_X = MinMaxScaler()
# ss_y = MinMaxScaler()

x_train = ss_X.fit_transform(x_train)

# x_train1 = ss_X.transform(x_train1)
# x_train2 = ss_X.transform(x_train2)
# x_train3 = ss_X.transform(x_train3)

# y_train = ss_y.fit_transform(y_train)
x_test = ss_X.transform(x_test)
x_valid = ss_X.transform(x_valid)

# # Random Forest
# rfr = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=20)
# rfr.fit(x_train, y_train)
# rfr_y_predict = ss_y.inverse_transform(rfr.predict(x_valid))
# print 'The MAPE value of Random Forest is', mean_absolute_percentage_error(y_valid, rfr_y_predict)
# y_rfr_predict = ss_y.inverse_transform(rfr.predict(x_test))
# make_submit('random_forest', y_rfr_predict)

# ExtraTrees Regressor
etr = ExtraTreesRegressor(random_state=seed, n_estimators=200, n_jobs=20, verbose=2)
etr.fit(x_train, y_train)
etr_y_predict = etr.predict(x_valid)
etr_train_predict = etr.predict(x_train)
print '(Valid) The MAPE value of Extra Tree is', mean_absolute_percentage_error(y_valid, etr_y_predict)
print '(train) The MAPE value of Extra Tree is', mean_absolute_percentage_error(y_train, etr_train_predict)
print 'Feature importance: ', etr.feature_importances_
y_etr_predict = etr.predict(x_test)
make_submit('extremely_randomized_trees', y_etr_predict)

# etr1 = ExtraTreesRegressor(random_state=seed, n_estimators=50, n_jobs=20, verbose=2)
# etr2 = ExtraTreesRegressor(random_state=seed, n_estimators=50, n_jobs=20, verbose=2)
# etr3 = ExtraTreesRegressor(random_state=seed, n_estimators=50, n_jobs=20, verbose=2)
# etr1.fit(x_train1, y_train1)
# print "etr1 done..."
# etr2.fit(x_train2, y_train2)
# print "etr2 done..."
# etr3.fit(x_train3, y_train3)
# print "etr3 done..."
#
# etr1_y_predict = etr1.predict(x_valid)
# etr2_y_predict = etr2.predict(x_valid)
# etr3_y_predict = etr3.predict(x_valid)
# y_val = list()
# for j, val_sample in enumerate(x_valid):
#     if val_sample[20] > 9000:
#         y_val.append(etr3_y_predict[j])
#     elif 9000 >= val_sample[20] > 5000:
#         y_val.append(etr2_y_predict[j])
#     else:
#         y_val.append(etr1_y_predict[j])
#
# y_val = np.asarray(y_val)
#
# print 'The MAPE value of Expert ETR is', mean_absolute_percentage_error(y_valid, y_val)
#
# y_etr1_predict = etr1.predict(x_test)
# y_etr2_predict = etr2.predict(x_test)
# y_etr3_predict = etr3.predict(x_test)
#
# y_predict = list()
# for i, test_sample in enumerate(x_test):
#     if test_sample[20] > 9000:
#         y_predict.append(y_etr3_predict[i])
#     elif 9000 >= test_sample[20] > 5000:
#         y_predict.append(y_etr2_predict[i])
#     else:
#         y_predict.append(y_etr1_predict[i])
#
# y_predict = np.asarray(y_predict)
#
# make_submit('expert_etr', y_predict)

# # KNN
# knr = KNeighborsRegressor(weights='distance', n_jobs=20)
# knr.fit(x_train, y_train)
# knr_y_predict = knr.predict(x_valid)
# print 'The MAPE value of KNN is', mean_absolute_percentage_error(y_valid, knr_y_predict)
# y_knr_predict = knr.predict(x_test)
# make_submit('knn', y_knr_predict)

# XGBoost
# xgbr = XGBRegressor(n_estimators=100,
#                     learning_rate=0.9,
#                     max_depth=9,
#                     min_child_weight=6,
#                     gamma=0,
#                     subsample=1,
#                     reg_alpha=0,
#                     reg_lambda=1,
#                     colsample_bytree=1,
#                     scale_pos_weight=1)
# xgbr.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=True)
# xgbr_y_predict = xgbr.predict(x_valid)
# print 'The MAPE value of XGBoost is', mean_absolute_percentage_error(y_valid, xgbr_y_predict)
# y_xgbr_predict = xgbr.predict(x_test)
# make_submit('xgboost', y_xgbr_predict)

# param_test = {
#     # 'max_depth': [9, 12, 15, 25, 30],
#     # 'min_child_weight': [2, 6, 8, 10]
#     'subsample': [i/10.0 for i in range(6, 10)],
#     'colsample_bytree': [i/10.0 for i in range(6, 10)]
# }
#
# gsearch1 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.9,
#                                                n_estimators=50,
#                                                max_depth=9,
#                                                min_child_weight=6),
#                         param_grid=param_test,
#                         n_jobs=20,
#                         verbose=2
#                         )
# gsearch1.fit(x_train, y_train)
# print gsearch1.best_params_
# print gsearch1.best_score_

# reg = list()
# reg.append(('standardize', StandardScaler()))
# reg.append(('etr', ExtraTreesRegressor()))
# pipeline = Pipeline(reg)
# parameters = {'etr__n_estimators': range(10, 110, 10),
#               'etr__max_features': ('auto', 'sqrt', 'log2', None),
#               'etr__max_depth': [10, 20, 50, 100, None]}
# gs = GridSearchCV(pipeline, parameters, verbose=2, refit=True, cv=20, n_jobs=-1)
# gs.fit(x_train, y_train)
# print gs.best_params_
# print gs.best_score_
#
# gs_y_predict = gs.predict(x_valid)
# print 'The MAPE value of Grid Search is', mean_absolute_percentage_error(y_valid, gs_y_predict)
#
# y_gs_predict = gs.predict(x_test)
# make_submit('grid_search', y_gs_predict)

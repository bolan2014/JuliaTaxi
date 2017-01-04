#-*- coding: utf-8 -*-

import numpy as np

def get_train(fname):
	data = np.loadtxt(fname, delimiter=',')

	#np.random.shuffle(data)

	X, y = (data.T[:-1]).T, data.T[-1]

	#print 'Data is ready ...'

	return X, y

def get_test(fname):
	data = np.loadtxt(fname, delimiter=',')
	return data

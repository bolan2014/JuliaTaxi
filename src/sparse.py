#-*- coding: utf-8 -*-

from random import randint

from util.distance import calcu_distance

#随机从下面四条轨迹中选取一条轨迹
def Roulette():
	return randint(1, 4)

if __name__ == '__main__':
	print Roulette()
	path = '/home/zzq/XPZ/'
	fname = '20140803_train_sorted.txt_unq'
	trackList = []
	fr = open(path + fname)
	for line in fr:
		trackList.append(line)
	fw = open(path + 'sparse_' + fname, 'w')
	i = 0
	while i < len(trackList):
		fw.write(trackList[i])
		i += Roulette()
	print 'Done'
	


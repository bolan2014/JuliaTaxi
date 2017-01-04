#-*- coding: utf-8 -*-

from random import randint
from util.distance import calcu_distance, trip_distance
from util.tripTime import calcu_time


#市中心坐标
base = (104.072227, 30.663456)

#四维（时段，到市中心距离，纬度，经度）
def start_end(record):
	res = []
	temp = record.strip().split('\t')
	res.append(int(temp[-1][-8:-6]))
	r_lat, r_lon = float(temp[1]), float(temp[2])
	res.append(calcu_distance(r_lon, r_lat, base[0], base[1]))
	res += [r_lat, r_lon]

	return res

#中途采样点（纬度，经度）二维
def middle_track(record):
	temp = record.strip().split('\t')
	return [temp[1], temp[2]]

def trip(tid, nums, tList):
	rst = [int(tid), len(tList)]

	rst += start_end(tList[0]) #起点
	rst += start_end(tList[-1]) #终点

	#中途采样
	gap = (len(tList)-2) / nums
	for i in range(1, nums+1):
		rst += middle_track(tList[gap*i])
	
	#旅程距离
	lat, lon, occupied = [], [], []
	for line in tList:
		temp = line.strip().split('\t')
		lat.append(float(temp[1]))
		lon.append(float(temp[2]))
		occupied.append(int(temp[3]))
	rst.append(trip_distance(lon, lat))

	#载客率
	rst.append(float(sum(occupied)) / len(occupied))

	#旅程时间
	s_time = tList[0].strip().split('\t')[-1][-8:]
	e_time = tList[-1].strip().split('\t')[-1][-8:]
	rst.append(calcu_time(s_time, e_time))

	return ','.join(map(str, rst)) # 2+4+4+10+1+1+1=23维
			

if __name__ == '__main__':
	path = '/home/zzq/XPZ/'
	fname = 'sparse_20140803_train_sorted.txt_unq'

	nums = 5 #每趟旅程中间采样的轨迹点数目

	taxiDict = {}

	fr = open(path + fname)
	fr.readline() #略过文件头

	#（出租车id，纬度，经度，载客，时间）
	pre = ''
	for line in fr:
		tmp = line.split('\t')
		if tmp[0] != pre:
			taxiDict[tmp[0]] = [line]
		else:
			taxiDict[tmp[0]].append(line)
		pre = tmp[0]
	print 'taxi dict is ready ...'

	fw = open('train.dat', 'w')

	for k in taxiDict:
		i, n = 0, len(taxiDict[k])
		while i < n:
			period = min(randint(11, 128), n-i)
			if period < 11:
				break
			fw.write(trip(k, nums, taxiDict[k][i:i+period]) + '\n')
			i += period
		print 'taxi', k, 'is ready ...'


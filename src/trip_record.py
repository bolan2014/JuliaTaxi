#-*- coding: utf-8 -*-

from random import randint
from util.distance import calcu_distance, trip_distance
from util.tripTime import calcu_time


#市中心坐标
base = (104.072227, 30.663456)

#四维（时段，到市中心距离，纬度，经度）
def start_end(record):
	res = []
	temp = record.strip().split(',')
	res.append(int(temp[-1][-8:-6]))
	r_lat, r_lon = float(temp[2]), float(temp[3])
	res.append(calcu_distance(r_lon, r_lat, base[0], base[1]))
	res += [r_lat, r_lon]

	return res

#中途采样点（纬度，经度）二维
def middle_track(record):
	temp = record.strip().split(',')
	return [temp[2], temp[3]]

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
		temp = line.strip().split(',')
		lat.append(float(temp[2]))
		lon.append(float(temp[3]))
		occupied.append(int(temp[4]))
	rst.append(trip_distance(lon, lat))

	#载客率
	rst.append(float(sum(occupied)) / len(occupied))

	#旅程时间
	#s_time = tList[0].strip().split(',')[-1][-8:]
	#e_time = tList[-1].strip().split(',')[-1][-8:]
	#rst.append(calcu_time(s_time, e_time))

	return ','.join(map(str, rst)) # 3+4+4+10+1+1=23维
			

if __name__ == '__main__':
	path = '/opt/exp_data/CurryKiller/replace/'
	fname = 'predPaths_test.txt'

	nums = 5 #每趟旅程中间采样的轨迹点数目

	tripDict = {}

	fr, fw = open(path + fname), open('test.dat', 'w')
	#fr.readline() #略过文件头

	#（旅程号，出租车id，纬度，经度，载客，时间）
	pre, tid, tripList = '', '', []
	for line in fr:
		tmp = line.split(',')
		if tmp[0] != pre:
			if tripList != []:
				fw.write(trip(tid, nums, tripList) + '\n')
				print 'taxi', tid, 'is ready ...'
				tripList = []
			else:
				tripList.append(line)
		else:
			tripList.append(line)
		pre, tid = tmp[0], tmp[1]
	fw.write(trip(tid, nums, tripList) + '\n')
	print 'taxi', tid, 'is ready ...'
	fw.close()


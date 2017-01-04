# -*- coding:utf-8 -*- 

import math

#计算两个GPS点之间距离
def calcu_distance(lon1, lat1, lon2, lat2):
	dx = lon1 - lon2
	dy = lat1 - lat2
	b = (lat1 + lat2) / 2.0;
	Lx = (dx/57.2958) * 6371004.0 * math.cos(b/57.2958)
	Ly = 6371004.0 * (dy/57.2958)
	return math.sqrt(Lx * Lx + Ly * Ly)

#计算旅程长度
def trip_distance(lonList, latList):
	if len(lonList) != len(latList):
		print '经纬度列表长度不匹配 ...'
	tDis = 0
	for i in range(len(lonList)-1):
		tDis += calcu_distance(lonList[i], latList[i], lonList[i+1], latList[i+1])
	return tDis

if __name__ == '__main__':
	A = (104.090826, 30.657726)
	B = (104.091938, 30.660658)
	C = (104.093317, 30.666393)

	lonList = [A[0], B[0], C[0]]
	latList = [A[1], B[1], C[1]]

	print calcu_distance(A[0], A[1], B[0], B[1])
	print calcu_distance(B[0], B[1], C[0], C[1])

	print trip_distance(lonList, latList)

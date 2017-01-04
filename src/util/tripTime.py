# -*- coding: utf-8 -*-

from datetime import datetime as dt

#把时间按小时分成12段
def time_int(ori_time):
    h = ori_time[0:2]
    return int(h)

#按日期把一个月分一下
def date_int(ori_date):
    m = ori_date[5:7]
    d = ori_date[8:10]
    return int(m), int(d)

#算时间间隔
def calcu_time(t1, t2):
	tFormat = '%H:%M:%S'
	a, b = dt.strptime(t1, tFormat), dt.strptime(t2, tFormat)
	return (b - a).seconds

if __name__ == '__main__':
	t1 = '01:02:03'
	t2 = '01:03:12'
	print calcu_time(t1, t2)

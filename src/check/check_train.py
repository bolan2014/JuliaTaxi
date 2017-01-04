import pandas as pd

base = '/opt/exp_data/CurryKiller/secondHalf/'

def check(test_file):
    df = pd.read_csv(test_file, header=None, names= \
            ['taxi_id','lat','lon','busy','dtime'])
	days = df.groupby
if __name__ == '__main__':
    fname = '20140830_train_sorted.txt'
    check(base + fname)

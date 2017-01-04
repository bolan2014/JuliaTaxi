import pandas as pd

base = '/opt/exp_data/CurryKiller/replace/'

def check(test_file):
    df = pd.read_csv(test_file, sep=',', header=None, names= \
            ['tripid','taxi_id','lat','lon','busy','dtime'])

    #trips = df.groupby('tripid')
    days = df.groupby('dtime')
    print days.size()
    #print 'count max:', max(trips.size())
    #print 'count min:', min(trips.size())
    #print 'count mean:', sum(trips.size()) / len(trips.size())

if __name__ == '__main__':
    fname = 'predPaths_test.txt'
    check(base + fname)

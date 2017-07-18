#!/usr/bin/env python

import sys

DELIM=","

def lastcolstats(filename):
    with open(filename) as f:
        lines = f.readlines()
        splitlines = [line.strip().split('\t') for line in lines if line.strip()]
    s = {}
    for sl in splitlines:
        lc = sl[len(sl)-1]
        if not lc in s:
            s[lc] = 0
        s[lc] += 1
    return s

def comparestats(stats1, stats2):
    keys = set([k1 for k1,v1 in stats1.items()] + [k2 for k2,v2 in stats2.items()])
    for key in keys:
        if key in stats1 and not key in stats2:
            print(stats1[key],DELIM,0,DELIM,key)
        elif key in stats2 and not key in stats1:
            print(0,DELIM,stats2[key],DELIM,key)
        else:
            print(stats1[key],DELIM,stats2[key],DELIM,key)

def usage():
    use="""
    Usage {0}
    Prints statistics related to final columns in two files
    Prints counts for each last column in each file
    If last column is not in one of the two files, there is a 0 count
    Takes two files as arguments
    Example:
    {0} dimsum16.train.tagger dimsum16.test.tagger
    """.format(sys.argv[0])
    print(use)

if __name__ == "__main__":
    try:
        fn1 = sys.argv[1]
        fn2 = sys.argv[2]
    except:
        usage()
        exit(1)
    stats1 = lastcolstats(fn1)
    stats2 = lastcolstats(fn2)
    print(fn1,DELIM,fn2,DELIM,"Jointtag")
    comparestats(stats1,stats2) # prints comparison

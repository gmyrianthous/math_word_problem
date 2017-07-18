#!/usr/bin/env python

import sys
import hashlib

with open(sys.argv[1]) as ds:
    lines = ds.readlines()
    rows = [row.strip().split('\t') for row in lines]
    for row in rows:
        if len(row) > 1:
            sha1 = hashlib.sha1()
            sha1.update(row[8].encode())
            row[8] = sha1.hexdigest() # digest for sentence ID
            row[7] = "" # empty supersense
            row[5] = '0' # no parent offset
            row[4] = 'O' # MWE always O
        print('\t'.join(row))


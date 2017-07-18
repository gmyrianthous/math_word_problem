#!/usr/bin/env python

import sys

###############################################################################
# convert dimsum file to tagger train file
###############################################################################
def usage():
    use="""
    Usage for {0}
    Supply a DiMSUM style sentence file to be converted to Tagger style input
    file for training. The script takes the MWE tags and the Supersense tags
    and combines them into a single joint tag separated by two underscores ("__").
    It prepends every line with it's corresponding word and appends the joint tag.

    Converted file printed to stdout.

    Example:
    {0} dimsum16.train > dimsum16.train.tagger
    """.format(sys.argv[0])
    print(use)

if len(sys.argv) != 2:
    print("Must supply an argument to convert.")
    usage()
    exit(1)

with open(sys.argv[1]) as dimsumfile:
    for line in dimsumfile:
        line = line.strip()
        line = line.split('\t')
        if len(line) > 8:
            word = line[1]
            jointtag = line[4]+"__"+line[7]
            line = [word] + line + [jointtag]
            line = '\t'.join(line)
            print(line)
        else:
            print()

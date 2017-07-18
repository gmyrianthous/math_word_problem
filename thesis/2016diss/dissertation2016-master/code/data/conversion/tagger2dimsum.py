#!/usr/bin/env python

import sys
sys.path.append('../../')
sys.path.append('../')
from dimsum.dimsumdataiterator import DimsumDataIterator
from dimsum.taggerdataiterator import TaggerDataIterator
from dimsum import tools

###############################################################################
# takes tagger output file and converts to dimsum style file 
###############################################################################

def usage():
    use = """
    Usage for {0}
    First argument is the DiMSUM test sentences used for predicting
    Second Argument is the Tagger predictions for the test sentences
    Will produce a DiMSUM style file printed to stdout with the predictions

    Example:
    {0} dimsum16.test.blind dimsum16.test.tagger.output.txt
    """.format(sys.argv[0])
    print(use)

try:
    test_sentences = [sentence for sentence in DimsumDataIterator(sys.argv[1])]
    pred_tags = [tags for tags in TaggerDataIterator(sys.argv[2])]
except:
    usage()
    exit()

for test_sent, preds in zip(test_sentences, pred_tags):
    dimsum_sent = tools.taggerpred2dimsumsent(test_sent, preds)
    for row in dimsum_sent:
        print('\t'.join(row))
    print() # newline separator

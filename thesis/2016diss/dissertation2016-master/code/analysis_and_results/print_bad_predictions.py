#!/usr/bin/env python

import sys
sys.path.append('../')
from dimsum.dimsumdataiterator import DimsumDataIterator
from dimsum import tools

def usage():
    script_name = sys.argv[0]
    usage_str = '''
    To use {0} put the gold standard dimsum filename followed by the predicted filename
    Any sentence that differs will be printed to stdout. Columns displayed are lemma,
    BIO tag sequence, offset sequence and Supersense tag sequence. gold appears before predicted
    in each column.

    Example: {0} dimsum16.test b1_baseline_predictions.csv
    '''.format(script_name)
    print(usage_str)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()
        exit()

    gold_filename = sys.argv[1]
    pred_filename = sys.argv[2]

    # make a dimsum data iterator for the training data
    gold_sentences_iter = DimsumDataIterator(gold_filename)
    gold_sentences = [sentence for sentence in gold_sentences_iter]

    # make a dimsum data iterator for the testing data
    pred_sentences_iter = DimsumDataIterator(pred_filename)
    pred_sentences = [sentence for sentence in pred_sentences_iter]

    i = 0
    e = 0
    for pred_sent, gold_sent in zip(pred_sentences, gold_sentences):
        lemma = tools.retrieveColumn(gold_sent, 2)
        pred_mwe = tools.retrieveColumn(pred_sent, 4)
        gold_mwe = tools.retrieveColumn(gold_sent, 4)
        pred_ss = tools.retrieveColumn(pred_sent, 7)
        gold_ss = tools.retrieveColumn(gold_sent, 7)
        if pred_mwe != gold_mwe or pred_ss != gold_ss:
            print("="*79)
            print("Sentence number",i)
            print("="*79)
            row = "{0:20} {1:10} {2:10} {3:15} {4:15}".format("lemma","gold mwe","pred mwe","gold ss","pred ss")
            print(row)
            print("-"*79)
            for l,gm,pm,gs,ps in zip(lemma,gold_mwe, pred_mwe,gold_ss,pred_ss):
                row = "{0:20} {1:10} {2:10} {3:15} {4:15}".format(l,gm,pm,gs,ps)
                print(row)
            print("\n")
            e += 1
        i+=1

    print("Total mismatched mwe or ss: ",e,"/",i)

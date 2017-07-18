#!/usr/bin/env python

import sys
sys.path.append('../')

from dimsum import tools
from dimsum import stats
from os.path import expanduser
from dimsum.dimsumdataiterator import DimsumDataIterator

def pprint_mwes(sentences_mwes):
    for sentence_mwes in training_sentences_mwes:
        BIOs,bios = sentence_mwes
        if BIOs:
            for BIO in BIOs:
                for row in BIO:
                    print(row)
                if BIO:
                    print()
        if bios:
            for bio in bios:
                for row in bio:
                    print(row)
                if bio:
                    print()

if __name__ == '__main__':
    # file location for dimsum data
    #home = expanduser("~")
    #path_prefix=home+"/data/DIMSUM/dimsum-data"
    #train_filename=path_prefix+"/dimsum16.train"
    if len(sys.argv) != 2:
        print("Must provide sentences file")
        print("Usage: {0} dimsum16.train".format(sys.argv[0]))
        exit()
    train_filename = sys.argv[1]
    training_sentences_iter = DimsumDataIterator(train_filename)
    training_sentences = [sentence for sentence in training_sentences_iter]

    training_sentences_mwes = tools.extractMWEs(training_sentences)

    qmwes = stats.quantityMWEs(training_sentences)
    qsentwmwes = stats.quantitySentWithMWE(training_sentences)
    avgmwelen = stats.avergeMWELength(training_sentences)
    qmwessht = stats.quantityMWESupersenseHeadTypes(training_sentences)
    qssht = stats.quantitySupersenseHeadTypes(training_sentences)
    qpostpmwessht = stats.quantitiesPOStagPerMWESupersenseHeadType(training_sentences)
    qss = stats.quantitySupersenses(training_sentences)

    print("="*79)
    print("DiMSIM Data Statistics Training Sentences")
    print("="*79)
    print("Quantity of Sentences:", len(training_sentences))
    print("Quantity of MWEs:", qmwes)
    print("Quantity of Supersenses:", sum([count for _,count in qss.items()]) )
    print("Quantity of Sentences with MWEs:", qsentwmwes)
    print("Average MWE Length: %.2f" % avgmwelen)
    print("Quantity of Supersense Head types (n./v.)")
    for headtype in ['n','v']:
        print(" ",headtype,qssht[headtype])
    print("Quantity of MWE Supersense Head types (n./v.)")
    for headtype in ['n','v']:
        print(" ",headtype,qmwessht[headtype])
    print("Quantities of POS tag heads per MWE Head type (n./v.)")
    for headtype in ['n','v']:
        for postag,count in qpostpmwessht[headtype].most_common():
            print(" ",headtype,":",postag,":",count)
    print("Quantities of Supersenses in all sentences combined")
    for supersense,count in qss.most_common():
        print(" ",supersense,":",count)

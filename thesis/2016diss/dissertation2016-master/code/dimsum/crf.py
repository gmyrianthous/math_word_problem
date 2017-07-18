#!/usr/bin/env python

import pycrfsuite
from dimsum import tools

###############################################################################
# class CRF
#
# wrapper class for crfsuite used in basline of DIMSUM task
###############################################################################
class CRF(object):
    # requires name for save file
    def __init__(self,savefile):
        # save file name for crfsuite
        self.savefile = savefile
        # feature indexes to extract
        self.featureidxs=[2,3]
        # label index to extract
        self.labelidx=4
        # context (forward/backward) to extract
        self.context=(-2,2)
        self.trainer = pycrfsuite.Trainer(verbose=False)
        self.tagger = pycrfsuite.Tagger()
        
    # add data to the trainer by extracting features and labels
    # uses the features indexes, context indexes and label index
    def add(self,sentences):
        # clear training data and save sentences, features, labels
        self.trainer.clear()
        self.sentences = []
        self.features = []
        self.labels = []
        for sentence in sentences:
            self.sentences.append(sentence)
            self.features.append(tools.sentenceToFeatures(sentence, featureidxs=self.featureidxs, context=self.context))
            self.labels.append(tools.sentenceToLabels(sentence, labelidx=self.labelidx))

        for feature,label in zip(self.features,self.labels):
            self.trainer.append(feature,label)
            
    # train CRF and save results to filename
    def train(self):
        self.trainer.train(self.savefile)

    # predict all items in X using the trained crf
    # load the saved data at prediction time
    def predict(self,sentences):
        self.tagger.open(self.savefile)
        features = []
        for sentence in sentences:
            features.append(tools.sentenceToFeatures(sentence, featureidxs=self.featureidxs, context=self.context))
        return [self.tagger.tag(feature) for feature in features]
        
    # set the indexes for the features to extract
    # a list of integers for columns in dimsum data
    def setFeatureIndexes(self,indexes):
        self.featureidxs = indexes

    # set the index for the labels
    # an integer for the column in dimsum data
    def setLabelIndex(self,index):
        self.labelidx = index

    # set the context limits tuple for feature extraction context
    # (forward/backward index limits)
    def setContext(self, context):
        self.context = context

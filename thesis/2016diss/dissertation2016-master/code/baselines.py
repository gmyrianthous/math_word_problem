#!/usr/bin/env python

from sklearn.metrics import classification_report
from dimsum.dimsumdataiterator import DimsumDataIterator
from dimsum import crf
from dimsum import tools
from os.path import expanduser
from os.path import exists
import itertools
import random
import math
import copy
import time

###############################################################################
# globals
###############################################################################
# file location for dimsum data
path_prefix="dimsum-data"
train_filename=path_prefix+"/dimsum16.train"
test_filename=path_prefix+"/dimsum16.test.blind"

pred_tab_csv_filename_prefix="predictions_baseline"
b1_pred_tab_csv_filename="b1_"+pred_tab_csv_filename_prefix+".csv"
b2_pred_tab_csv_filename="b2_"+pred_tab_csv_filename_prefix+".csv"
b3_pred_tab_csv_filename="b3_"+pred_tab_csv_filename_prefix+".csv"

###############################################################################
# def baseline_report
#
# make report for correct labels and predictions
###############################################################################
def baseline_report(correct_labels, predicted_labels):
    y_true = [c for c in itertools.chain(*correct_labels)]
    y_pred = [p for p in itertools.chain(*predicted_labels)]
    
    report = classification_report(y_true,y_pred)
    
    return report

###############################################################################
# def hold_out_validation
#
# train, predict and run simple hold out validation for evaluation of model crf
# sentences : Sentences to validate with CRF
# crf : Un-trained CRF to make predictions (parameters set)
# p : Percent to hold out for validation (value 0-1)
###############################################################################
def hold_out_validation(sentences, crf, p=0.4):
    print(" running simple hold-out validation with",str(p*100),"percent held out")
    sent_len = len(sentences)
    # number of sentences to hold out
    n = math.floor(sent_len*p)
    # shuffle sentences to randomise
    shuf_sent = copy.deepcopy(sentences)
    random.shuffle(shuf_sent)

    test_sentences = shuf_sent[n:]
    holdout_sentences = shuf_sent[:n]

    old_savefile = crf.savefile
    crf.savefile = "/tmp/hovalid.crfsuite."+str(time.time())+".tmp"
    crf.add(test_sentences)
    crf.train()
    predictions = crf.predict(holdout_sentences)
    correct = [tools.sentenceToLabels(sentence, labelidx=crf.labelidx) for sentence in holdout_sentences]
    print(" Hold-out validation results:")
    print(baseline_report(correct, predictions))
    crf.savefile = old_savefile
    crf.trainer.clear()

###############################################################################
# def b1_predict_dimsum_sentences
#
# Baseline number 1:
# Uses 2 CRFS: 
#   1. Features Training lemma+POS, Predict MWE Tags
#   2. Features Training lemma+POS+MWE, Predict Supersense
# Combine two predictions with testing sentences
# Systematically produce parent offset sequence from MWE tags
# 
# return list of predicted sentences
###############################################################################
def b1_predict_dimsum_sentences(training_sentences, testing_sentences, forceRetrain=False):
    mwe_crf_savefile = "b1_mwe_crf.baseline.crfsuite"
    supersense_crf_savefile = "b1_supersense_crf.baseline.crfsuite"

    if forceRetrain:
        print(" Force retrain set")
    ##########
    # Make the inital CRF for MWE tags prediction (BIO tags)
    ##########
    print(" Making CRF for MWE tags (BIO tags)")
    # make CRF with savefile name
    mwe_crf = crf.CRF(mwe_crf_savefile)
    # set feature indexes, label index, and context limits in CRF
    print("  setting features to Lemma and POS tags")
    mwe_crf.setFeatureIndexes([2,3]) #lemma+POS indexes
    print("  setting label to MWE tags (BIO tags)")
    mwe_crf.setLabelIndex(4) # MWE tag label index
    print("  setting feature context limits")
    mwe_crf.setContext((-2,2)) # feature context limits

    # run hold-out validation for MWE tags
    #print("\nRunning Hold-Out Validation Tests for MWE tagging using training sentences from", train_filename)
    #hold_out_validation(training_sentences, mwe_crf)

    # add all sentences to CRF for baseline
    if forceRetrain or not exists(mwe_crf_savefile):
        print(" Extracting features and adding training sentences to CRF for MWE tags (BIO tags)")
        mwe_crf.add(training_sentences)
        # run training on baseline
        print(" Training CRF for MWE tags (BIO tags)")
        mwe_crf.train()
    else:
        print(" CRFSUITE savefile exists for CRF for MWE tags, no training needed.")
    # run prediction
    print(" Predicting MWE tags with test sentences using CRF for MWE tags (BIO tags)")
    mwe_predictions = mwe_crf.predict(testing_sentences)

    ##########
    # Create second CRF for Supersense tags prediction 
    ##########
    print(" Making CRF for Supersense tags")
    supersense_crf = crf.CRF(supersense_crf_savefile)
    print("  setting features to Lemma, POS and MWE tags")
    supersense_crf.setFeatureIndexes([2,3,4]) #lemma+POS+BIO indexes
    print("  setting label to Supersense tags")
    supersense_crf.setLabelIndex(7) # Supersense tag label index
    print("  setting feature context limits")
    supersense_crf.setContext((-2,2)) # feature context limits

    # run hold-out validation for Supersense tags
    #print("\nRunning Hold-Out Validation Tests for Supersense tagging using training sentences from", train_filename)
    #hold_out_validation(training_sentences, supersense_crf)

    #train second CRF with training data using lemma+POS+BIO tags as features
    if forceRetrain or not exists(supersense_crf_savefile):
        print(" Extracting features and adding training sentences to CRF for Supersense tags")
        supersense_crf.add(training_sentences)
        print(" Training CRF for Supersense tags")
        supersense_crf.train()
    else:
        print(" CRFSUITE savefile exists for CRF for Supersense tags, no training needed.")
    
    #use predicted output of mwe_crf + testing sentences
    #for testing in the second crf
    print(" Making new test sentences with previous MWE tag predictions")
    testing_sentences_plus_mwe_predictions = []
    for i in range(len(testing_sentences)):
        testing_sentences_plus_mwe_predictions.append(tools.replaceSentenceColumn(testing_sentences[i], mwe_predictions[i], 4))
    print(" Predicting Supersense tags with updated test sentences using CRF for Supersense tags")
    supersense_predictions = supersense_crf.predict(testing_sentences_plus_mwe_predictions)

    # check mwes/supsersenses and "fix" them if there are errors
    for i in range(len(supersense_predictions)):
        supersense_prediction = supersense_predictions[i]
        mwe_prediction = mwe_predictions[i]
        if not tools.isValidMWESequence(mwe_prediction):
            mwe_predictions[i] = tools.fixInvalidMWESequence(mwe_prediction)
            mwe_prediction = mwe_predictions[i]
            testing_sentences_plus_mwe_predictions[i]=tools.replaceSentenceColumn(testing_sentences[i], mwe_predictions[i], 4)
            if not tools.isValidMWESequence(mwe_prediction):
                print("MWE STILL NOT FIXED IN BASELINE 1")
        # not valid if MWE for supersense, fix
        if not tools.isValidSupersenseSequence(mwe_prediction, supersense_prediction):
            # if there is not a head in the MWE sequence, fix invalid supersense sequence
            # move first supersense prediction to MWE head (naive assumption)
            supersense_predictions[i] = tools.fixAllInvalidSupersenseSequences(mwe_prediction, supersense_prediction)
            if not tools.isValidSupersenseSequence(mwe_prediction, supersense_predictions[i]):
                print("SS STILL NOT FIXED IN BASELINE 1")

    ###########
    # Add new supersense predictions with previous MWE tag predictions
    # Run function to deterministically write indexes for MWE tag parent offsets
    ###########
    final_predicted_sentences = []
    for i in range(len(testing_sentences_plus_mwe_predictions)):
        # add supersense predictions to testing sentences with mwe predictions
        pred_sent = tools.replaceSentenceColumn(testing_sentences_plus_mwe_predictions[i], supersense_predictions[i], 7)
        # get the MWE tag sequence from the predicted sentence
        mwe_tag_seq = tools.retrieveColumn(pred_sent, 4)
        # make the parent offset sequence for the MWE tag sequence
        parent_offset_seq = tools.makeParentOffsetColumn(mwe_tag_seq)
        # replace the parent offset sequence column in the predicted sentence
        pred_sent = tools.replaceSentenceColumn(pred_sent, parent_offset_seq, 5)
        # add the final predicted sentence with MWE tags, Supersense tags and Parent Offset Sequence
        final_predicted_sentences.append(pred_sent)

    return final_predicted_sentences

###############################################################################
# def b2_predict_dimsum_sentences
#
# Baseline number 2:
# Almost the same as baseline 1 but 2nd 
# Uses 2 CRFS: 
#   1. Features Training lemma+POS, Predict MWE Tags
#   2. Features Training lemma+POS, Predict Supersense
# Combine two predictions with testing sentences
# Systematically produce parent offset sequence from MWE tags
#
# return list of predicted sentences
###############################################################################
def b2_predict_dimsum_sentences(training_sentences, testing_sentences, forceRetrain=False):
    mwe_crf_savefile = "b2_mwe_crf.baseline.crfsuite"
    supersense_crf_savefile = "b2_supersense_crf.baseline.crfsuite"
    
    if forceRetrain:
        print(" Force retrain set")
    ##########
    # Make the inital CRF for MWE tags prediction (BIO tags)
    ##########
    print(" Making CRF for MWE tags (BIO tags)")
    # make CRF with savefile name
    mwe_crf = crf.CRF(mwe_crf_savefile)
    # set feature indexes, label index, and context limits in CRF
    print("  setting features to Lemma and POS tags")
    mwe_crf.setFeatureIndexes([2,3]) #lemma+POS indexes
    print("  setting label to MWE tags (BIO tags)")
    mwe_crf.setLabelIndex(4) # MWE tag label index
    print("  setting feature context limits")
    mwe_crf.setContext((-2,2)) # feature context limits

    # add all sentences to CRF for baseline
    if forceRetrain or not exists(mwe_crf_savefile):
        print(" Extracting features and adding training sentences to CRF for MWE tags (BIO tags)")
        mwe_crf.add(training_sentences)
        # run training on baseline
        print(" Training CRF for MWE tags (BIO tags)")
        mwe_crf.train()
    else:
        print(" CRFSUITE savefile exists for CRF for MWE tags, no training needed.")
    # run prediction
    print(" Predicting MWE tags with test sentences using CRF for MWE tags (BIO tags)")
    mwe_predictions = mwe_crf.predict(testing_sentences)

    ##########
    # Create second CRF for Supersense tags prediction 
    ##########
    print(" Making CRF for Supersense tags")
    supersense_crf = crf.CRF(supersense_crf_savefile)
    print("  setting features to Lemma, POS and MWE tags")
    supersense_crf.setFeatureIndexes([2,3]) #lemma+POS indexes
    print("  setting label to Supersense tags")
    supersense_crf.setLabelIndex(7) # Supersense tag label index
    print("  setting feature context limits")
    supersense_crf.setContext((-2,2)) # feature context limits

    #train second CRF with training data using lemma+POS tags as features
    if forceRetrain or not exists(supersense_crf_savefile):
        print(" Extracting features and adding training sentences to CRF for Supersense tags")
        supersense_crf.add(training_sentences)
        print(" Training CRF for Supersense tags")
        supersense_crf.train()
    else:
        print(" CRFSUITE savefile exists for CRF for Supersense tags, no training needed.")
    
    #use predicted output of mwe_crf + testing sentences
    #for testing in the second crf
    print(" Making new test sentences with previous MWE tag predictions")
    testing_sentences_plus_mwe_predictions = []
    for i in range(len(testing_sentences)):
        testing_sentences_plus_mwe_predictions.append(tools.replaceSentenceColumn(testing_sentences[i], mwe_predictions[i], 4))
    print(" Predicting Supersense tags with updated test sentences using CRF for Supersense tags")
    supersense_predictions = supersense_crf.predict(testing_sentences_plus_mwe_predictions)

    # check mwes/supsersenses and "fix" them if there are errors
    for i in range(len(supersense_predictions)):
        supersense_prediction = supersense_predictions[i]
        mwe_prediction = mwe_predictions[i]
        if not tools.isValidMWESequence(mwe_prediction):
            mwe_predictions[i] = tools.fixInvalidMWESequence(mwe_prediction)
            mwe_prediction = mwe_predictions[i]
            testing_sentences_plus_mwe_predictions[i]=tools.replaceSentenceColumn(testing_sentences[i], mwe_predictions[i], 4)
            if not tools.isValidMWESequence(mwe_prediction):
                print("MWE STILL NOT FIXED IN BASELINE 2")
        # not valid if MWE for supersense, fix
        if not tools.isValidSupersenseSequence(mwe_prediction, supersense_prediction):
            # if there is not a head in the MWE sequence, fix invalid supersense sequence
            # move first supersense prediction to MWE head (naive assumption)
            supersense_predictions[i] = tools.fixAllInvalidSupersenseSequences(mwe_prediction, supersense_prediction)
            if not tools.isValidSupersenseSequence(mwe_prediction, supersense_predictions[i]):
                print("SS STILL NOT FIXED IN BASELINE 2")

    ###########
    # Add new supersense predictions with previous MWE tag predictions
    # Run function to deterministically write indexes for MWE tag parent offsets
    ###########
    final_predicted_sentences = []
    for i in range(len(testing_sentences_plus_mwe_predictions)):
        # add supersense predictions to testing sentences with mwe predictions
        pred_sent = tools.replaceSentenceColumn(testing_sentences_plus_mwe_predictions[i], supersense_predictions[i], 7)
        # get the MWE tag sequence from the predicted sentence
        mwe_tag_seq = tools.retrieveColumn(pred_sent, 4)
        # make the parent offset sequence for the MWE tag sequence
        parent_offset_seq = tools.makeParentOffsetColumn(mwe_tag_seq)
        # replace the parent offset sequence column in the predicted sentence
        pred_sent = tools.replaceSentenceColumn(pred_sent, parent_offset_seq, 5)
        # add the final predicted sentence with MWE tags, Supersense tags and Parent Offset Sequence
        final_predicted_sentences.append(pred_sent)

    return final_predicted_sentences

###############################################################################
# def b3_predict_dimsum_sentences
#
# Baseline number 3:
# Uses 1 CRF: 
#   1. Features Training lemma+POS, Predict MWE+Supersense Tags together (concatenated)
# Split concatenated prediction to two tags (MWE/Supersense),
# Combine two predictions with testing sentences
# Systematically produce parent offset sequence from MWE tags
#
# return list of predicted sentences
###############################################################################
def b3_predict_dimsum_sentences(training_sentences, testing_sentences, forceRetrain=False):
    b3_crf_savefile = "b3_crf.baseline.crfsuite"
    
    if forceRetrain:
        print(" Force retrain set")
    ##########
    # Make the inital for MWE+Supersense tags prediction
    ##########
    print(" Making CRF for MWE+Supersense tags")
    # make CRF with savefile name
    b3_crf = crf.CRF(b3_crf_savefile)
    # set feature indexes, label index, and context limits in CRF
    print("  setting features to Lemma and POS tags")
    b3_crf.setFeatureIndexes([2,3]) #lemma+POS indexes
    print("  setting label to MWE+Supersense tags")
    b3_crf.setLabelIndex([4,7]) # MWE+Supersense tag label index
    print("  setting feature context limits")
    b3_crf.setContext((-2,2)) # feature context limits

    # add all sentences to CRF for baseline
    if forceRetrain or not exists(b3_crf_savefile):
        print(" Extracting features and adding training sentences to CRF")
        b3_crf.add(training_sentences)
        # run training on baseline
        print(" Training CRF")
        b3_crf.train()
    else:
        print(" CRFSUITE savefile exists for CRF, no training needed.")
    # run prediction
    print(" Predicting MWE tags with test sentences using CRF")
    b3_concat_predictions = b3_crf.predict(testing_sentences)
    mwe_predictions = []
    supersense_predictions = []
    for concat_pred in b3_concat_predictions:
        # temp mwe and ss predition sequence lists
        mwe_pred = []
        ss_pred = []
        # split the tags and make separate mwe seq and ss seq
        for mwetag___sstag in concat_pred:
            mwetag,sstag = mwetag___sstag.split("___")
            mwe_pred.append(mwetag)
            ss_pred.append(sstag)

        # add this prediction pair to their lists
        mwe_predictions.append(mwe_pred)
        supersense_predictions.append(ss_pred)
        
    # replace the sentence columns in the testing sentences with the mwe predictions
    print(" Making new test sentences with previous MWE tag predictions")
    testing_sentences_plus_mwe_predictions = []
    for i in range(len(testing_sentences)):
        testing_sentences_plus_mwe_predictions.append(tools.replaceSentenceColumn(testing_sentences[i], mwe_predictions[i], 4))

    # check mwes/supsersenses and "fix" them if there are errors
    for i in range(len(supersense_predictions)):
        supersense_prediction = supersense_predictions[i]
        mwe_prediction = mwe_predictions[i]
        if not tools.isValidMWESequence(mwe_prediction):
            mwe_predictions[i] = tools.fixInvalidMWESequence(mwe_prediction)
            mwe_prediction = mwe_predictions[i]
            # mwe prediction sequence was updated, update the testing sentences
            testing_sentences_plus_mwe_predictions[i]=tools.replaceSentenceColumn(testing_sentences[i], mwe_predictions[i], 4)
            if not tools.isValidMWESequence(mwe_prediction):
                print("MWE STILL NOT FIXED IN BASELINE 3")
        # not valid if MWE for supersense, fix
        if not tools.isValidSupersenseSequence(mwe_prediction, supersense_prediction):
            # if there is not a head in the MWE sequence, fix invalid supersense sequence
            # move first supersense prediction to MWE head (naive assumption)
            supersense_predictions[i] = tools.fixAllInvalidSupersenseSequences(mwe_prediction, supersense_prediction)
            if not tools.isValidSupersenseSequence(mwe_prediction, supersense_predictions[i]):
                print("SS STILL NOT FIXED IN BASELINE 3")

    ###########
    # Add supersense predictions with previous MWE tag predictions
    # Run function to deterministically write indexes for MWE tag parent offsets
    ###########
    final_predicted_sentences = []
    for i in range(len(testing_sentences_plus_mwe_predictions)):
        # add supersense predictions to testing sentences with mwe predictions
        pred_sent = tools.replaceSentenceColumn(testing_sentences_plus_mwe_predictions[i], supersense_predictions[i], 7)
        # get the MWE tag sequence from the predicted sentence
        mwe_tag_seq = tools.retrieveColumn(pred_sent, 4)
        # make the parent offset sequence for the MWE tag sequence
        parent_offset_seq = tools.makeParentOffsetColumn(mwe_tag_seq)
        # replace the parent offset sequence column in the predicted sentence
        pred_sent = tools.replaceSentenceColumn(pred_sent, parent_offset_seq, 5)
        # add the final predicted sentence with MWE tags, Supersense tags and Parent Offset Sequence
        final_predicted_sentences.append(pred_sent)

    return final_predicted_sentences

###############################################################################
# def kfold_predict_dimsum_sentences
#
# run k predictions by folding training sentences
# evaluation done externally by provided dimsumeval.py
###############################################################################
def kfold_predict_dimsum_sentences(training_sentences, baseline_function, K=10):
    n = math.ceil(len(training_sentences)/K) # num of items in K folds
    pred_kfolds = [] # predictions for each k
    # split sentences into k-folds
    kfolds = [training_sentences[i:i+n] for i in range(0, len(training_sentences), n)]
    # for each fold, get testing/training data. predict and save.
    for i in range(len(kfolds)):
        k =  i+1
        print(" Splitting fold",k,"and saving predicted sentences")
        # save test fold
        ktest = kfolds[i]
        # save train fold
        ktrain = [unfolded for unfolded in itertools.chain(*kfolds[:i] + kfolds[i+1:])]
        # run predictions for this fold and save results
        pred_kfolds.append(baseline_function(ktrain, ktest, forceRetrain=True))
    # return all predicted folds
    return (kfolds,pred_kfolds)

###############################################################################
# run __main__ program
###############################################################################
if __name__ == "__main__":
    # write header to baseline output
    print('=' * 79)
    print(":: RUNNING BASELINE SYSTEMS ::")
    print('=' * 79, '\n')

    ##########
    # Make the inital data from the training and testing data
    ##########
    print(":: Making training sentence data from file", train_filename,"::")
    # make a dimsum data iterator for the training data
    training_sentences_iter = DimsumDataIterator(train_filename)
    training_sentences = [sentence for sentence in training_sentences_iter]
    print(":: Making testing sentence data from file", test_filename,"::")
    # make a dimsum data iterator for the testing data
    testing_sentences_iter = DimsumDataIterator(test_filename)
    testing_sentences = [sentence for sentence in testing_sentences_iter]

    ###########
    # Predict the sentences for the training and testing data
    ###########
    print(":: Running predictions for training and testing files ::")
    print(":: Baseline 1 ::")
    b1_final_pred_sent = b1_predict_dimsum_sentences(training_sentences, testing_sentences, forceRetrain=True)
    print(":: Baseline 2 ::")
    b2_final_pred_sent = b2_predict_dimsum_sentences(training_sentences, testing_sentences, forceRetrain=True)
    print(":: Baseline 3 ::")
    b3_final_pred_sent = b3_predict_dimsum_sentences(training_sentences, testing_sentences, forceRetrain=True)
    
    ###########
    # Write completed predictions to file for test
    ###########
    print(":: Writing results for baseline to", b1_pred_tab_csv_filename,"::")
    tools.sentencesToTabbedCsv(b1_final_pred_sent, b1_pred_tab_csv_filename)
    
    print(":: Writing results for baseline to", b2_pred_tab_csv_filename,"::")
    tools.sentencesToTabbedCsv(b2_final_pred_sent, b2_pred_tab_csv_filename)
    
    print(":: Writing results for baseline to", b3_pred_tab_csv_filename,"::")
    tools.sentencesToTabbedCsv(b3_final_pred_sent, b3_pred_tab_csv_filename)

    ###########
    # Write folded predictions to files
    ###########
    # k-fold data to evaluate externally
    # fold data k times and return results
    # output files to ./results directory
    print(":: Running K-Fold tests using training sentences and exporting to files ::")
    # for each prediction function, run kfold cross-evaluation
    for baseline_pred_function,baseline_prefix in [(b1_predict_dimsum_sentences,"b1_"),(b2_predict_dimsum_sentences,"b2_"),(b3_predict_dimsum_sentences,"b3_")]:
        print(" Using prediction function {0}".format(str(baseline_pred_function)))
        # get all gold and predicted folds 
        (gold_kfolds,pred_kfolds) = kfold_predict_dimsum_sentences(training_sentences, baseline_pred_function, K=10)
        # write the fold results to files to evaluate externally with dimsumeval script
        for i in range(len(gold_kfolds)):
            k = i+1
            kgold_filename = "results/"+baseline_prefix+"baseline.gold.fold"+str(k)+".csv"
            print(" Writing results for gold fold",k,"to", kgold_filename)
            tools.sentencesToTabbedCsv(gold_kfolds[i], kgold_filename)
            kpred_filename = "results/"+baseline_prefix+"baseline.pred.fold"+str(k)+".csv"
            print(" Writing results for pred fold",k,"to", kpred_filename)
            tools.sentencesToTabbedCsv(pred_kfolds[i], kpred_filename)

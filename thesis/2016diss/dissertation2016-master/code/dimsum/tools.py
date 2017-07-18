#!/usr/bin/env python

from dimsum import DATA_COLUMNS
import copy
import re

# extract features from each index in sentence
def sentenceToFeatures(sentence, featureidxs, context):
    return [extractFeature(sentence, index, featureidxs, context) for index in range(len(sentence))]

# extract label from each index in sentence
def sentenceToLabels(sentence, labelidx):
    if isinstance(labelidx, list):
        # there is a list of indexed
        # concatenate labels as if they were one item
        label_lists = []
        for idx in labelidx:
            label_lists.append([extractLabel(sentence, index, idx) for index in range(len(sentence))])
        labels = []
        for zipped in zip(*label_lists):
            labels.append('___'.join(map(str, zipped)))
        return labels
    else:
        return [extractLabel(sentence, index, labelidx) for index in range(len(sentence))]

# extract feature for a given index of a sentence
# the feature idxs list are the indexes for the features to extract in each column
# the context is the forward and backward min/max indexes for to extract (None otherwise)
def extractFeature(sentence, index, featureidxs, context):
    #features = ['bias'] #inital bias term
    features = []
    for featureidx in featureidxs:
        feature = DATA_COLUMNS[featureidx] + "=" + sentence[index][featureidx]
        features.append(feature)
    # lower limit context exists, extract previous features and append
    if context[0]:
        # go backwards and get previous context feature
        for i in range(context[0],0):
            contextidx=index+i
            # only extract previous context if available (i.e. gt 0 index)
            if contextidx >= 0:
                # for each feature index, extract the feature in the previous contect
                for featureidx in featureidxs:
                    feature = str(i) + ":" + DATA_COLUMNS[featureidx] + "=" + sentence[contextidx][featureidx]
                    features.append(feature)
    # upper limit context exists, extract proceeding features and append
    if context[1]:
        # go forward and get next context feature
        for i in range(1,(context[1]+1)):
            contextidx=index+i
            # only extract proceeding context if available (i.e. lt sentence length)
            if contextidx < len(sentence):
                # for each feature index, extract the feature in the proceeding contect
                for featureidx in featureidxs:
                    feature = str(i) + ":" + DATA_COLUMNS[featureidx] + "=" + sentence[contextidx][featureidx]
                    features.append(feature)    
    return features

# extact the label for the sentence at the index
# the column given by labelidx is the label
def extractLabel(sentence, index, labelidx):
    return sentence[index][labelidx]

# replace the column in the sentence with the new column
# the index is the column in the sentence to replace
def replaceSentenceColumn(sentence, newcolumn, index):
    if len(sentence) != len(newcolumn):
        print("ERROR: new column and sentence different lengths")
        return None
    # ensure sentence data not changed by deep copy
    newsentence = copy.deepcopy(sentence)
    for i in range(len(newsentence)):
        # get row in sentence
        row = newsentence[i]
        # replace index with the ith column entry
        row[index] = newcolumn[i]
        # save the new row
        newsentence[i] = row
    return newsentence

# retrieve a single column from the sentence data
def retrieveColumn(sentence, index):
    return [row[index] for row in sentence]

# write sentences to file in CoNLL style
# same used for training and testing data in DiMSUM task
def sentencesToTabbedCsv(sentences, csvfilename):
    with open(csvfilename,'w+') as csvfile:
        for sentence in sentences:
            for row in sentence:
                tabbedrow = "\t".join(map(str, row)) + "\n"
                csvfile.write(tabbedrow)
            csvfile.write("\n")

# deterministically create a parent offset column for the MWE tag column (BIO tags)
# inital tests show 0% error on usage
def makeParentOffsetColumn(mwecolumn):
    parentoffsetcolumn = []
    bio_stack = []
    BIO_stack = []
    for i in range(len(mwecolumn)):
        offset = i+1
        mwetag = mwecolumn[i]
        if mwetag == 'B':
            parentoffsetcolumn.append(0)
            BIO_stack.append(offset)
        elif mwetag == 'b':
            parentoffsetcolumn.append(0)
            bio_stack.append(offset)
        elif mwetag == 'I':
            try:
                top = BIO_stack.pop()
            except:
                top = 0
            parentoffsetcolumn.append(top)
            BIO_stack.append(offset)
        elif mwetag == 'i':
            try:
                top = bio_stack.pop()
            except:
                top = 0
            parentoffsetcolumn.append(top)
            bio_stack.append(offset)
        elif mwetag == 'O':
            parentoffsetcolumn.append(0)
            BIO_stack = []
        elif mwetag == 'o':
            parentoffsetcolumn.append(0)
            bio_stack = []
    return [stringOffset for stringOffset in map(str, parentoffsetcolumn)]

# extract all MWEs from all sentences
# for each sentence extract list of MWEs
def extractMWEs(sentences):
    mwes = []
    for sentence in sentences:
        mwes.append(extractMWE(sentence))
    return mwes

# push BIO bio sequence onto stack and append to list
# then return list of MWE sequences for this sentence
def extractMWE(sentence):
    BIO_stack = []
    BIO = []
    bio_stack = []
    bio = []
    for row in sentence:
        mwe_tag = row[4]
        if mwe_tag == 'B':
            if BIO: # previous MWE exists, start new one
                BIO_stack.append(BIO)
                BIO = []
            BIO.append(row)
        elif mwe_tag == 'I':
            BIO.append(row)
        elif mwe_tag == 'O':
            if BIO:
                BIO_stack.append(BIO)
            BIO = []
        elif mwe_tag == 'b':
            if bio: # previous MWE exists, start new one
                bio_stack.append(bio)
                bio = []
            bio.append(row)
        elif mwe_tag == 'i':
            bio.append(row)
        elif mwe_tag == 'o':
            if bio:
                bio_stack.append(bio)
            bio = []

    return (BIO_stack, bio_stack)

# check for valid MWE sequence
def isValidMWESequence(sequence):
    # RegEx pattern taken from dimsumeval.py
    pat=r'^(O|B(o|b[iīĩ]+|[IĪĨ])*[IĪĨ]+)+$'
    mwe_string = ''.join(sequence)
    return re.search(pat, mwe_string) != None

# very dumb sequence fixer
# just makes MWEs all Os (non-mwe)
def fixInvalidMWESequence(sequence):
    return ['O' for tag in sequence]

# just a patchy solution to sequences that are invalid
# just a methodology to prove that naive unfounded choices
# perform worse then predicted ones (not very important?)
# like saying a horrible solution is better than good one? (obvious?)
def fixInvalidMWESequence2(sequence):
    hasb = False
    hasB = False
    newtag = None
    fixed = []
    for tag in sequence:
        if tag == 'i' and not hasb:
            if hasB:
                tag = 'O' #default
            else:
                newtag  = 'B'
                hasB = True
        elif (tag == 'I' or tag == 'o') and not hasB:
            newtag = 'B'
            hasB = True
        elif tag == 'O':
            hasb = False
            hasB = False
        elif tag == 'B':
            hasB = True
        elif tag == 'b':
            hasb = True
            hasB = False

        if newtag:
            # fixed tags in list, replace top
            if fixed: 
                fixed[len(fixed)-1] = newtag
            else:
                #empty list, replace tag with new tag
                tag = newtag
        fixed.append(tag)
        newtag = None
    # patchwork frankenstein boooo, not a good idea, not justified
    # check last tag, if it is a weak mwe, end in I to make valid sequence
    lasti = len(fixed)-1
    lasttag = fixed[lasti]
    if lasttag == 'b' or lasttag == 'o' or lasttag == 'i':
        fixed[lasti] = 'I'
    return fixed

# invalid Supersense sequence if I or i has supersense
# means strong/weak mwe heads have empty or additional tag == bad!
def isValidSupersenseSequence(mweseq, supersenseseq):
    if len(mweseq) != len(supersenseseq):
        return False
    else:
        for i in range(len(mweseq)):
            mwetag = mweseq[i]
            supersensetag = supersenseseq[i]
            if (mwetag == 'I' or mwetag == 'i') and supersensetag != "":
                return False
        return True

# fixes invalid Supersense sequence given an MWE sequence
# If I or i has supersense, move to previous Bb if not populated
# otherwise just remove supersense (set to empty string)
# assume only one MWE sequence to fix
def fixSingleInvalidSupersenseSequence(mweseq, supersenseseq):
    if len(mweseq) != len(supersenseseq):
        print("ERROR: MWE Sequence is not the same Length as Supersense sequence, cannot fix")
        return supersenseseq
    else:
        newssseq = copy.deepcopy(supersenseseq)
        # assume there is only one mwe, so there should be only a single B or b
        mweidx = [i for i,x in enumerate(mweseq) if x == 'B' or x == 'b']
        if len(mweidx) != 1:
            print("WARNING: MWE Sequence does not contain weak or strong MWE, cannot fix")
            return supersenseseq
        else:
            # confirm, only single MWEs here
            mweidx = mweidx.pop()
            # if there is a not a supersense at this index, we need to fix it
            if not supersenseseq[mweidx]:
                # find the first supersense on a MWE I or i, then replace at mweidx
                supersenseidxs = [i for i,x in enumerate(mweseq) if (x == 'I' or x == 'i') and (supersenseseq[i] != '')]
                if not supersenseidxs: # no supersenses, that is ok
                    # no supersenses, empty supersense sequence, therefore "correct"
                    return supersenseseq
                else: # supersense(s) found, use the first by default
                    ssidx = supersenseidxs[0]
                    # replace MWE head with proceeding supersense
                    newssseq[mweidx] = newssseq[ssidx]

            # next remove all other supersenses at I or i
            for i in range(len(supersenseseq)):
                if mweseq[i] == 'I' or mweseq[i] == 'i':
                    newssseq[i] = ''

        # if we got here, sequence changed, return fixed sequence
        return newssseq

# split mweseq and supersenseseq into single mwes/ss pairs and fix, then return
def fixAllInvalidSupersenseSequences(mweseq, supersenseseq):
    mweidxs = [i for i,x in enumerate(mweseq) if x == 'B' or x == 'b']
    if not mweidxs:
        return supersenseseq
    elif len(mweidxs) == 1:
        newssseq = fixSingleInvalidSupersenseSequence(mweseq, supersenseseq)
        return newssseq
    else:
        # prefix of inital sequence, eveything leading up to first mwe
        newssseq = copy.deepcopy(supersenseseq)
        newssseq = newssseq[0:mweidxs[0]]
        # greater than 1
        mweidxs = mweidxs + [len(supersenseseq)]
        # make start,end index tuples for mwe indexes
        for start,end in zip(mweidxs[:-1],mweidxs[1:]):
            # split each sub mwe and supersense sequences
            submweseq = mweseq[start:end]
            subsupersenseseq = supersenseseq[start:end]
            # concatenate the "fixed" sub sequence
            newssseq = newssseq + fixSingleInvalidSupersenseSequence(submweseq, subsupersenseseq)
        # return the concatenated fixed supersense sequence
        return newssseq

# take tagger predictions and change them into a dimsum style
def taggerevalpreds2dimsumpreds(taggerpreds):
    split_preds = []
    split_pred = []
    for taggerpred in taggerpreds:
        if taggerpred:
            split_pred.append(taggerpred.split())
        else:
            split_preds.append(split_pred)
            split_pred = []
    dimsum_preds = []
    for split_pred in split_preds:
        # last item in prediction row is joint tag prediction
        jointtags = [col[-1].split('__') for col in split_pred]
        # MWE sequence
        mwes = retrieveColumn(jointtags,0)
        if not isValidMWESequence(mwes):
            #print("WARNING: Invalid tagger MWE sequence, replacing")
            mwes = fixInvalidMWESequence(mwes)
        # Supersense sequence
        ss = retrieveColumn(jointtags,1)
        # Parent offset column sequence
        poc = makeParentOffsetColumn(mwes)
        # prefixes for predictions: [offset,word,lemma,pos]
        pres = [pre[1:5] for pre in split_pred]
        # empty sequence to fill row 7,9
        es = ['']*len(split_pred)
        # make single dimsum style prediction by adding first 4 columns
        # and appending MWEs, Parent Offsets, Empty sequence, Supersenses, Empty sequence
        dimsum_pred = []
        for i in range(len(split_pred)):
            dimsum_pred.append(pres[i] + [mwes[i]] + [poc[i]] + [es[i]] + [ss[i]] + [es[i]])
        dimsum_preds.append(dimsum_pred)
    return dimsum_preds

def taggerpred2dimsumsent(dimsum_sent, tagger_pred):
    pred_mwes = retrieveColumn(tagger_pred,1)
    validMWE = isValidMWESequence(pred_mwes)
    if not validMWE:
        pred_mwes = fixInvalidMWESequence(pred_mwes)
    pred_ss = retrieveColumn(tagger_pred,2)
    parent_offset_seq = makeParentOffsetColumn(pred_mwes)
    pred_sent = copy.deepcopy(dimsum_sent)
    pred_sent = replaceSentenceColumn(pred_sent, parent_offset_seq, 5)
    pred_sent = replaceSentenceColumn(pred_sent, pred_mwes, 4)
    pred_sent = replaceSentenceColumn(pred_sent, pred_ss, 7)
    return pred_sent

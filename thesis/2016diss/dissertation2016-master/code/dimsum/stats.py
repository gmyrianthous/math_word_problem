#!/usr/bin/env python

from dimsum import DATA_COLUMNS
from dimsum import tools
from collections import Counter

# Quantity of MWEs
def quantityMWEs(sentences):
    n = 0
    mwes = tools.extractMWEs(sentences)
    for mwe in mwes:
        outter,inner = mwe
        if outter:
            n += len(outter)
        if inner:
            n += len(inner)
    return n
    
# Quantity sentences have MWEs
def quantitySentWithMWE(sentences):
    n = 0
    mwes = tools.extractMWEs(sentences)
    for i in range(len(sentences)):
        outter,inner = mwes[i]
        if outter or inner:
            n += 1
    return n

# Average MWE length (BIO length)
def avergeMWELength(sentences):
    mwes = tools.extractMWEs(sentences)
    lengths = []
    for mwe in mwes:
        outter,inner = mwe
        if outter:
            lengths.append(len(outter))
        if inner:
            lengths.append(len(inner))
    return sum(lengths)/len(lengths)
        
# Quantity v./n. MWEs
def quantityMWESupersenseHeadTypes(sentences):
    counter = Counter()
    mwes = tools.extractMWEs(sentences)
    for mwe in mwes:
        outters,inners = mwe
        if outters:
            for outter in outters:
                for row in outter:
                    supersense = row[7]
                    if supersense:
                        counter[supersense.split('.')[0]]+=1
        if inners:
            for inner in inners:
                for row in inner:
                    supersense = row[7]
                    if supersense:
                        counter[supersense.split('.')[0]]+=1
    return counter

# Quantity v./n. for all supersenses
def quantitySupersenseHeadTypes(sentences):
    counter = Counter()
    for sentence in sentences:
        for row in sentence:
            supersense = row[7]
            if supersense:
                counter[supersense.split('.')[0]]+=1
    return counter
    
# Count POS tag for mwe supersenses v./n.
def quantitiesPOStagPerMWESupersenseHeadType(sentences):
    counter = Counter()
    counter['v'] = Counter()
    counter['n'] = Counter()
    mwes = tools.extractMWEs(sentences)
    for mwe in mwes:
        outters,inners = mwe
        if outters:
            for outter in outters:
                for row in outter:
                    supersense = row[7]
                    if supersense:
                        headtype = supersense.split('.')[0]
                        postag = row[3]
                        counter[headtype][postag]+=1
        if inners:
            for inner in inners:
                for row in inner:
                    supersense = row[7]
                    if supersense:
                        headtype = supersense.split('.')[0]
                        postag = row[3]
                        counter[headtype][postag]+=1
    return counter

# count the quantity of each supersense
def quantitySupersenses(sentences):
    counter = Counter()
    for sentence in sentences:
        for row in sentence:
            supersense = row[7]
            if supersense:
                counter[supersense] += 1
    return counter

#!/usr/bin/end python

###############################################################################
# class DimsumData
#
# opens and iterates over sentence data in a files provided by task 10 for
# ACL '16 Detecting minimal semantic units and their meaning DIMSUM
###############################################################################
class DimsumDataIterator(object):
    def __init__(self,filename):
        # all the lines from the file
        self.lines = open(filename).readlines()
        # clean the newlines, make each line a list
        self.lines = [line.strip().split('\t') for line in self.lines]
        # start index of next data item
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        item = self.__nextItem()
        if item:
            return item
        else:
            raise StopIteration()
    # for ease of use, just calls __next__
    def next(self):
        return self.__next__()
    # get the next item. list of lists. each list is a row in sentence,
    # with the corresponding columns for the shared task. (in README.md)
    def __nextItem(self):
        if self.i < len(self.lines):
            start = self.i
            end = start
            # move to next empty line
            while self.lines[end] != ['']:
                end+=1
            # set i to next start
            self.i = end+1
            # split lines from start to end (single sentence)
            item = self.lines[start:end]
            # return a item, the sentence in tuples for each line
            return item
        else:
            return None


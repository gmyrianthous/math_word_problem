import json 
import re
import string
from collections import OrderedDict, Counter
import nltk
import random 
import itertools
import sys 
from random import shuffle
from tqdm import tqdm
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger
from nltk.parse.stanford import StanfordDependencyParser

# Returns: list of dicts
def read_data_json(filename):
	print("Loading data..")
	with open(filename) as data_file:    
	    data = json.load(data_file)	
	#print("\t-Number of training examples: " + str(len(data)))

	return data

# Report number of problems, sentences, words; vocab size; mean words and sentences per problem.
def report_dataset_stats(data):
	number_of_problems = len(data)
	number_of_sentences = 0
	number_of_words = 0
	vocabulary = set([])

	for i in range(0, len(data)):
		number_of_sentences += len(re.split(r'[.!?]+', data[i]['sQuestion'])) - 1
		words = re.findall(r'\w+', data[i]['sQuestion'])
		for word in words:
			vocabulary.add(word)

		number_of_words += len(words)

	vocab_size = len(vocabulary)
	mean_words_per_problem = number_of_words / number_of_problems
	mean_sentences_per_problem = number_of_sentences / number_of_problems

	print("Dataset stats")
	print("\t -Number of problems: " + str(number_of_problems))
	print("\t -Number of sentences: " + str(number_of_sentences))
	print("\t -Number of words: " + str(number_of_words))
	print("\t -Vocabulary size: " + str(vocab_size))
	print("\t -Mean words per problem: " + str(mean_words_per_problem))
	print("\t -Mean sentences per problem: " + str(mean_sentences_per_problem) + "\n\n")


if __name__ == "__main__":
	# Fix random seed so that results are reproducible. 
	random.seed(26)

	# Read the data
	data = read_data_json('data/SingleOp.json')

	# Report dataset's stats -> stdout
	report_dataset_stats(data)

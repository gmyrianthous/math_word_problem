import json 
from pprint import pprint
import nltk
from nltk.parse.stanford import StanfordDependencyParser
import re
import numpy

# Paths for NLTK models
path_to_jar = 'stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar' 
path_to_models_jar = 'stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0-models.jar'


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


# Get unique templates (including numbers)
# Generic template: x = ((a op b) op c)
def get_templates(data):
	templates = set([])
	for i in range (0, len(data)):
		templates.add(data[i]['lEquations'][0])

	print(templates)
	print(len(templates))


# Incomplete - for future use. 
def dependecny_parsing():
	dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

	print(data[1]['sQuestion'])
	result = dependency_parser.raw_parse(data[0]['sQuestion'].lstrip())
	#result = dependency_parser.raw_parse("    Hello World")

	dep = result.__next__() 
	print(list(dep.triples()))

# Construct a transition system 
# Accepts: a problem/question (including alignment, )
# Returns: an equation/template
def transition_system(problem):
	question = problem['sQuestion']
	print("Question: " + question)

	alignment = problem['lAlignments']
	print("Alignments: " + str(alignment))

	# Get the numbers pointed by alignments.
	numbers = []  
	for index in alignment:
		number = ""
		number += question[index]
		for i in range(index+1, len(question)):
			if question[i].isdigit():
				number += question[i]
			else:
				break
		numbers.append(int(number))

	print("Numbers extracted from alignments: " + str(numbers))

	matrix = numpy.zeros((len(numbers), len(numbers)))

	for i in range(0, matrix.shape[0]):
		matrix[i][i] = 1
	print(matrix)




if __name__ == '__main__':

	# Read the data
	data = read_data_json("data/SingleOP.json")
	#pprint(data)

	# Report dataset stats
	report_dataset_stats(data)

	#get_templates(data)

	transition_system(data[1])


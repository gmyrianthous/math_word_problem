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

# Get the numbers that appear in the template. 
def get_numbers_in_template(quantities, alignment):
	numbers = []
	for i in range(0, len(quantities)):
		numbers.append(quantities[i]['val'])

	# Put the numbers in the correct order, using the alignment
	numbers_ordered = []

	for al in alignment:
		numbers_ordered.append(numbers[al])

	return numbers_ordered

# Extract features for problems that belong to the AllArith dataset
# that has a different format. 
def extract_features(data, indices):
	id_features_dict = OrderedDict()

	for math_problem in data:
		if math_problem['iIndex'] in indices:
			features = Counter()

			question = math_problem['sQuestion']
			index = math_problem['iIndex']
			quantities = math_problem['quantities'] # list of dicts
			alignment = math_problem['lAlignments']
			equation = math_problem['lEquations'][0]
			solution = math_problem['lSolutions'][0]

			# Get the number of quantities that appear in the text
			number_of_quantities = len(quantities)

			# Get the numbers in the correct order
			numbers = get_numbers_in_template(quantities, alignment)



			print("Q:" + question)
			print(quantities)
			print(alignment)
			print(solution)
			print(equation)
			print(number_of_quantities)
			print("Extracted numbers: " + str(numbers))

	sys.exit(1)




	return id_features_dict
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

def get_operations_in_template(equation):
	operations = []
	for char in equation:
		if char == "+":
			operations.append("+")
		elif char == "-":
			operations.append("-")
		elif char == "*":
			operations.append("*")
		elif char == "/":
			operations.append("/")
	return operations	

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

			# Extract the operations as they appear in the equation
			operations = get_operations_in_template(equation)

			# Feature 1: previousWord:word_operation_-
			# Get previous word
			previous_words_temp = []

			for i in range(0, len(quantities)):
				curr_index = quantities[i]['start']-2

				if question[curr_index] == " ":
					curr_index -= 1

				reversed_word = ""
				curr_char = question[curr_index]
				while curr_char != " " and curr_index >= 0:
					reversed_word += curr_char
					curr_index -= 1
					curr_char = question[curr_index]
				previous_words_temp.append(re.sub('[^A-Za-z0-9]+', '', reversed_word[::-1]).lower())

			# Put the words in the correct order
			previous_words = []
			for order in alignment:
				previous_words.append(previous_words_temp[order])

			# For the first number that appears in the template the sign is always positive
			features['previousWord:'+str(previous_words[0])+"_operation:+"] += 1
			for i in range(0, len(operations)):
				features['previousWord:'+str(previous_words[i+1])+"_operation:"+operations[i]] += 1


			# Feature 2: nextWord:word_operation_-
			next_words_temp = []
			for i in range(0, len(quantities)):
				curr_index = quantities[i]['end']+1

				word = ""	
				curr_char = question[curr_index]
				while curr_char != " " and curr_index < len(question)-1:
					word += curr_char
					curr_index += 1
					curr_char = question[curr_index]		
				next_words_temp.append(re.sub('[^A-Za-z0-9]+', '', word).lower())

			# Put the words in the correct order
			next_words = []
			for order in alignment:
				next_words.append(next_words_temp[order])

			# For the first number that appears in the template the sign is always positive
			features['nextWord:'+str(next_words[0])+"_operation:+"] += 1
			for i in range(0, len(operations)):
				features['nextWord:'+str(next_words[i+1])+"_operation:"+operations[i]] += 1



			# Feature 3: eachFlag:True/False_operation:op
			if ' each' in question:
				features['questionContainsEach:True_lastOperation:'+operations[-1]] += 1

			# Feature 4: operation:op_positionInTemplate:pos
			for i in range(0, len(operations)):
				features['operation:'+operations[i]+"_positionInTemplate:"+str(i+1)] += 1

			# Feature 5: secondOperation in template
			if len(operations) > 1:
				features['secondOperation:'+operations[1]] += 1

			# Feature 6: asksHowMany:True/False_resultType:int/float
			if 'how many' in question:
				if (solution).is_integer():
					features['asksHowMany:True_resultType:int'] += 1
				else:
					features['asksHowMany:True_resultType:float'] +=1 
			else:
				if (solution).is_integer():
					features['asksHowMany:False_resultType:int'] += 1
				else:
					features['asksHowMany:False_resultType:float'] += 1

			# Feature 7: asksHowMuch:True/False_resultType:int/float
			if 'how much' in question:
				if (solution).is_integer():
					features['asksHowMuch:True_resultType:int'] += 1
				else:
					features['asksHowMuch:True_resultType:float'] +=1 
			else:
				if (solution).is_integer():
					features['asksHowMuch:False_resultType:int'] += 1
				else:
					features['asksHowMuch:False_resultType:float'] += 1			

			# Add the features into the dictionary
			id_features_dict[index] = {}
			id_features_dict[index]['features'] = features

			# Also add some info that will be useful for the next actions
			id_features_dict[index]['question'] = question
			id_features_dict[index]['alignment'] = alignment
			id_features_dict[index]['equation'] = equation
			id_features_dict[index]['solution'] = solution
			id_features_dict[index]['numbers'] = numbers
			id_features_dict[index]['operations'] = operations
			id_features_dict[index]['previousWords'] = previous_words
			id_features_dict[index]['nextWords'] = next_words

	return id_features_dict

#def extract_combination_features(problem, combination):

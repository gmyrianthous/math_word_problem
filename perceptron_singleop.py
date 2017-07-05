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
from sklearn.model_selection import KFold

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

# SingleOp: 6-fold cross validation
def k_fold_cross_validation(data, dataset = "SingleOp"):
	kf = KFold(n_splits=2)
	kf.get_n_splits(data)

	k_folds = []
	for trainining_indices, testing_indices in kf.split(data):
		k_folds.append((trainining_indices, testing_indices))

	final_data = []
	for i in range(0, len(k_folds)):
		training_data = []
		testing_data = []

		training_set = k_folds[i][0]
		testing_set = k_folds[i][1]

		for index in training_set:
			training_data.append(data[index])

		for index in testing_set:
			testing_data.append(data[index])

		final_data.append((training_data, testing_data))

	return final_data

def get_numbers_in_template(alignment, question):
	numbers = []  
	for index in alignment:
		number = ""
		number += question[index]
		for i in range(index+1, len(question)):
			if question[i].isdigit():
				number += question[i]
			elif question[i] == "." and question[i+1].isdigit():
				number += question[i]
			else:
				break
		numbers.append(number)
	return numbers

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

# Extract features for a given set of data points.
def extract_features(data):
	id_features_dict = OrderedDict()

	for math_problem in data:
		features = Counter()

		# Extract problem's info
		index = math_problem['iIndex']
		question = math_problem['sQuestion']
		alignment = math_problem['lAlignments']
		equation = math_problem['lEquations'][0]
		solution = math_problem['lSolutions'][0]

		# Extract the numbers (as appear in the template)
		numbers = get_numbers_in_template(alignment, question)

		# Extract the operations as they appear in the equation
		operations = get_operations_in_template(equation)

		# We need to bring the templates into a 'global form'. 
		# Equations that follow the a op (b op c) should be transformed into (a op b) op c
		# The alignment and the extraxted numbers should also be modified in order to match the new template.
		#if "))" in equation:
			#print("modified")
		#	equation, alignment, numbers, operations = convert_template(equation, alignment, numbers, operations)


		# Feature 1: previousWord:word_operation_-
		# Get previous word
		previous_words = []
		for i in range(0, len(alignment)):
			index2 = alignment[i]-2
			reversed_word = ""
			curr_char = question[index2]
			while curr_char != " ":
				reversed_word += curr_char
				index2 -= 1
				curr_char = question[index2] 

			previous_words.append(re.sub('[^A-Za-z0-9]+', '', reversed_word[::-1]).lower())

		# For the first number that appears in the template the sign is always positive
		features['previousWord:'+str(previous_words[0])+"_operation:+"] += 1
		features['previousWord:'+str(previous_words[1])+"_operation:"+operations[0]] += 1


		# Feature 2: nextWord:word_operation_-
		# Get next word
		next_words = []
		for i in range(0, len(alignment)):
			index3 = alignment[i] + len(str(numbers[i])) + 1
			word = ""
			curr_char = question[index3]
			while curr_char != " " and curr_char != "?":
				word += curr_char
				index3 += 1
				curr_char = question[index3]
			next_words.append(re.sub('[^A-Za-z0-9]+', '', word).lower())

		# For the first number that appears in the template the sign is always positive
		features['nextWord:'+str(next_words[0])+"_operation:+"] += 1
		features['nextWord:'+str(next_words[1])+"_operation:"+operations[0]] += 1


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
		#id_features_dict[index]['previousWords2'] = previous_words_2
		#id_features_dict[index]['nextWords2'] = next_words_2
	return id_features_dict

# Initialise the feature weights
def initialise_weights(data):
	weights = Counter()

	for example in data:
		features = data[example]['features']
		for feature in features:
			weights[feature] = 0

	return weights

# Function for shuffling an OrderedDict
def shuffle_data(data):
    # Shuffle the keys
    keys = list(data.keys())
    random.shuffle(keys)

    # Write the shuffled keys in another dictionary.
    shuffled_dictionary = OrderedDict()
    for key in keys:
        shuffled_dictionary[key] = data[key]

    return shuffled_dictionary


def get_equation_combinations():
	symbol = ['a', 'b']
	op = ['+', '-', '/', '*']
	combinations = []

	for symbols in itertools.permutations(symbol):
	    for ops in itertools.product(op, repeat=1):
	        combinations.append("%s %s %s" % (
	            symbols[0], ops[0], symbols[1]))

	return combinations

# Extract combination features
def extract_combination_features(problem, combination):
	features = Counter()

	alignment = problem['alignment']
	numbers = problem['numbers']
	question = problem['question']
	solution = problem['solution']

	# Get the operations of the combination
	operations = get_operations_in_template(combination)

	# Feature 1: previous word for each number
	previous_words = problem['previousWords']

	# For the first number that appears in the template the sign is always positive
	features['previousWord:'+str(previous_words[0])+"_operation:+"] += 1
	features['previousWord:'+str(previous_words[1])+"_operation:"+operations[0]] += 1

	# Feature 2: next word for each number
	next_words = problem['nextWords']

	# For the first number that appears in the template the sign is always positive
	features['nextWord:'+str(next_words[0])+"_operation:+"] += 1
	features['nextWord:'+str(next_words[1])+"_operation:"+operations[0]] += 1


	return features	

# Function to compute the dot product between a combination's feature list and the weight vector
def dot_product(features_dict, weights):
	dot = 0
	for feature in features_dict.keys():
		if feature in weights.keys():
			dot += features_dict[feature] * weights[feature]
	return dot

# Get the most promising combination 
def argmax(problem, weights):
	equation = problem['equation']
	numbers = problem['numbers']
	question = problem['question']

	# Get the possible combinations
	combinations = get_equation_combinations()

	# Construct each combination and compute the features and score for each. 
	max_dot = None 
	max_features = None
	max_combination = None

	for combination in combinations:

		# Fill in current combination
		combination_filled_in = ""

		for char in combination:
			if char == 'a':
				combination_filled_in += str(numbers[0])
			elif char == 'b':
				combination_filled_in += str(numbers[1])
			else:
				combination_filled_in += char	

		# Extract features for the current combination
		combination_features = extract_combination_features(problem, combination_filled_in)


		combination_result = eval(combination_filled_in)

		#if isinstance( combination_result, int) or (combination_result).is_integer():
		dot = dot_product(combination_features, weights)
		if (max_dot is None or dot > max_dot):
			max_dot = dot
			max_features = combination_features
			max_combination = combination_filled_in

	return max_dot, max_features, max_combination

# Train the strucutred perceptron and learn the weights. 
# Multiple passes, shuffline and averaging can improve the performance of our classifier. 
def train(data, iterations=9, debugging=False):
	# Initialise the weights
	weights = initialise_weights(data)

	# Multiple passes
	for i in tqdm(range(iterations)):
		training_accuracy = 0
		# Shuffling
		data = shuffle_data(data)

		for problem in data:

			# Correct features of the problem. 
			features = data[problem]['features']

			# Predict the sequence of numbers and operations according to the template. 
			y_hat_dot, y_hat_features, y_hat_combination = argmax(data[problem], weights)

			# Extract the correct/wrong features
			features_for_addition = []
			features_for_subtraction = []

			for feature in features:
				if feature not in y_hat_features.keys():
					features_for_addition.append(feature)

			for feature in y_hat_features.keys():
				if feature not in features.keys() or feature not in weights.keys():
					features_for_subtraction.append(feature)

			# Update the weights according to the prediction
			for feature in features_for_addition:
				weights[feature] += 1

			for feature in features_for_subtraction:
				weights[feature] -= 1
			
			# Execute the predicted equation
			prediction = eval(y_hat_combination)
			solution = data[problem]['solution']

			if solution == prediction:
				training_accuracy += 1
		#print("Training accuracy: " + str(training_accuracy / len(data) * 100))

	return weights

# Testing phase of structured perceptron
def test(data, weights, debugging=False):
	correct_counter = 0

	for problem in data:
		features = data[problem]['features']
		solution = data[problem]['solution']
		question = data[problem]['question']
		equation = data[problem]['equation']

		# Predict the sequence of numbers and operations according to the template. 
		y_hat_dot, y_hat_features, y_hat_combination = argmax(data[problem], weights)	

		#print("Question: " + question)
		#print("Correct features: " + str(features))
		#print("Predicted features: " + str(y_hat_features))
		#print("Correct solution: " + str(equation))
		#print("Predicted solution: " + y_hat_combination)
		#print("")

		# Execute the predicted equation
		prediction = eval(y_hat_combination)

		if solution == prediction:
			correct_counter += 1

	accuracy = correct_counter / len(data) * 100
	#print ("Testing accuracy: " + str(accuracy))

	return accuracy

if __name__ == "__main__":
	# Fix random seed so that results are reproducible. 
	random.seed(26)

	# Read the data
	data = read_data_json('data/SingleOp.json')

	# Report dataset's stats -> stdout
	report_dataset_stats(data)

	# Perform 2-fold cross validation -> list of 6 tuples(training, testing)
	two_folds = k_fold_cross_validation(data)	

	accuracy_per_fold = []
	accuracy = 0 
	for i in range(0, len(two_folds)):
		print("Fold "+ str(i+1) + "/"+str(len(two_folds)))
		training_data = two_folds[i][0]
		testing_data = two_folds[i][1]

		# Extract features for training and testing examples
		training_features = extract_features(training_data)
		testing_features = extract_features(testing_data)

		feature_weights = train(training_features)

		# Test the perceptron
		curr_accuracy = test(testing_features, feature_weights)
		accuracy_per_fold.append(curr_accuracy)
		accuracy += curr_accuracy 

	print("Testing accuracy: " + str(accuracy / len(accuracy_per_fold)))
	print("Accuracy per fold: " + str(accuracy_per_fold))
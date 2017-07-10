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

# Paths for NLTK models
path_to_jar = 'stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar' 
path_to_models_jar = 'stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0-models.jar'

# Returns: list of dicts
def read_data_json(filename):
	with open(filename) as data_file:    
	    data = json.load(data_file)	
	print("Dataset has been loaded.")

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


def handle_command_line():
	dataset_name = ""
	if len(sys.argv) != 2:
		print("Please provide a dataset name (MultiArith, SingleOp, AllArith) and try again!")
		sys.exit(1)
	else:
		if sys.argv[1] != "MultiArith" and sys.argv[1] != "SingleOp" and sys.argv[1] != 'AllArith':
			print("Please provide the correct dataset name!")
			sys.exit(1)
		else:
			dataset_name = sys.argv[1]
	return dataset_name

def get_dataset_indices(data):
	indices = []
	for problem in data:
		indices.append(int(problem['iIndex']))
	return indices

def read_folds(directory, dataset_indices):
	with open(directory) as f:
		testing_indices = []
		training_indices = []
		for line in f:
			testing_indices.append(int(line))
		for index in dataset_indices:
			if index not in testing_indices:
				training_indices.append(index)

	return (training_indices, testing_indices)

# MultiArith: 6-fold cross validation
# SingleOp: 5-fold cross validation
def k_fold_cross_validation(data, dataset):
	dataset_indices = get_dataset_indices(data)

	# list of tuple(training, testing)
	k_folds = []
	if dataset == "MultiArith":
		for i in range(0, 6):
			k_folds.append(read_folds('data/commoncore/fold'+str(i)+'.txt', dataset_indices))
	elif dataset == "SingleOp":
		for i in range(0, 5):
			k_folds.append(read_folds('data/illinois/fold'+str(i)+'.txt', dataset_indices))
	return k_folds

def get_numbers_in_template(alignment, question):
	numbers = []  
	for index in alignment:
		number = ""
		number += question[index]
		for i in range(index+1, len(question)):
			if question[i].isdigit() or question[i]==".":
				number += question[i]
			else:
				break

		if number[len(number)-1] == ".":
			number = number[:-1]

		if isinstance(eval(number), int):
			numbers.append(int(number))
		elif isinstance(eval(number), float):
			numbers.append(float(number))
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

# Transform a op (b op c) to (a op b) op c
def convert_template(equation, alignment, numbers, operations):

	numbers_new = [numbers[1], numbers[2], numbers[0]]
	alignment_new = [alignment[1], alignment[2], alignment[0]]
	operations_new = operations[::-1]
	equation_new = "X=(("+str(float(numbers_new[0]))+operations_new[0]+str(float(numbers_new[1]))+")"+operations_new[1]+str(float(numbers_new[2]))+")"

	return equation_new, alignment_new, numbers_new, operations_new

# Extract features for a given set of data points.
def extract_features(data, indices):
	# Syntactic dependecy parser (Stanford corenlp)
	dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

	id_features_dict = OrderedDict()

	for math_problem in data:
		if math_problem['iIndex'] in indices:
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
			if "))" in equation:
				#print("modified")
				equation, alignment, numbers, operations = convert_template(equation, alignment, numbers, operations)


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
			for i in range(0, len(operations)):
				features['previousWord:'+str(previous_words[i+1])+"_operation:"+operations[i]] += 1


			# Feature 2: nextWord:word_operation_-
			# Get next word
			next_words = []
			for i in range(0, len(alignment)):
				index3 = alignment[i] + len(str(numbers[i])) + 1
				word = ""
				curr_char = question[index3]
				while curr_char != " " and index3 < len(question)-1:
					word += curr_char
					index3 += 1
					curr_char = question[index3]
				next_words.append(re.sub('[^A-Za-z0-9]+', '', word).lower())

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
			#features['operation:'+operations[1]+"_positionInTemplate:2"] += 1

			# Feature 5: secondOperation in template
			if len(operations) > 1:
				features['secondOperation:'+operations[1]] += 1

			print(question)
			print(equation)
			print(numbers)
			print(features)
			print("")

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

# Function that produces all the possible combinations of the template
# Total: 96 operations
def get_equation_combinations(dataset_name):
	if dataset_name == "MultiArith":
		symbol = ['a', 'b', 'c']
		op = ['+', '-', '/', '*']
		combinations = []

		for symbols in itertools.permutations(symbol):
			for ops in itertools.product(op, repeat=2):
				combinations.append("(%s %s %s) %s %s" % (
					symbols[0], ops[0], symbols[1], ops[1], symbols[2]))
	elif dataset_name == "SingleOp":
		symbol = ['a', 'b']
		op = ['+', '-', '/', '*']
		combinations = []

		for symbols in itertools.permutations(symbol):
		    for ops in itertools.product(op, repeat=1):
		        combinations.append("%s %s %s" % (
		            symbols[0], ops[0], symbols[1]))		

	return combinations

# Function to compute the dot product between a combination's feature list and the weight vector
def dot_product(features_dict, weights):
	dot = 0
	for feature in features_dict.keys():
		if feature in weights.keys():
			dot += features_dict[feature] * weights[feature]
	return dot

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
	for i in range(0, len(operations)):
		features['previousWord:'+str(previous_words[i+1])+"_operation:"+operations[i]] += 1

	# Feature 2: next word for each number
	next_words = problem['nextWords']

	# For the first number that appears in the template the sign is always positive
	features['nextWord:'+str(next_words[0])+"_operation:+"] += 1
	for i in range(0, len(operations)):
		features['nextWord:'+str(next_words[i+1])+"_operation:"+operations[i]] += 1

	# Feature 3: eachFlag:True/False_lastOperation:op
	if ' each' in question:
		features['questionContainsEach:True_lastOperation:'+operations[-1]] += 1
	
	# Feature 4: operation:op_positionInTemplate:pos
	for i in range(0, len(operations)):
		features['operation:'+operations[i]+"_positionInTemplate:"+str(i+1)] += 1
	#features['operation:'+operations[1]+"_positionInTemplate:2"] += 1

	# Feature 5: secondOperation in template
	if len(operations) > 1:
		features['secondOperation:'+operations[1]] += 1

	return features	


# Get the most promising combination 
def argmax(problem, weights, dataset_name):
	equation = problem['equation']
	numbers = problem['numbers']
	question = problem['question']

	# Get the possible combinations
	combinations = get_equation_combinations(dataset_name)

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
			elif char == 'c':
				combination_filled_in += str(numbers[2])
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
def train(data, dataset_name, iterations=9,  debugging=False):
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
			y_hat_dot, y_hat_features, y_hat_combination = argmax(data[problem], weights, dataset_name)

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
def test(data, weights, dataset_name, debugging=False):
	correct_counter = 0

	for problem in data:
		features = data[problem]['features']
		solution = data[problem]['solution']
		question = data[problem]['question']
		equation = data[problem]['equation']

		# Predict the sequence of numbers and operations according to the template. 
		y_hat_dot, y_hat_features, y_hat_combination = argmax(data[problem], weights, dataset_name)	

		# Execute the predicted equation
		prediction = eval(y_hat_combination)

		if solution == prediction:
			correct_counter += 1

	accuracy = correct_counter / len(data) * 100

	return accuracy

if __name__ == "__main__":
	# Fix random seed so that results are reproducible. 
	random.seed(26)
 
	dataset_name = handle_command_line()

	# Read the data
	data = read_data_json('data/'+dataset_name+'.json')

	# Report dataset's stats -> stdout
	report_dataset_stats(data)

	# Perform k-fold cross validation -> list of k tuples(training, testing)
	k_folds = k_fold_cross_validation(data, dataset_name)

	accuracy_per_fold = []
	accuracy = 0 
	for i in range(0, len(k_folds)):
		print("Fold "+ str(i+1) + "/"+str(len(k_folds)))
		training_indices = k_folds[i][0]
		testing_indices = k_folds[i][1]

		# Extract features for training and testing examples
		training_features = extract_features(data, training_indices)
		testing_features = extract_features(data, testing_indices)

		# Train the structured perceptron in order to learn the weights
		feature_weights = train(training_features, dataset_name)

		# Test the perceptron
		curr_accuracy = test(testing_features, feature_weights, dataset_name)
		accuracy_per_fold.append(curr_accuracy)
		accuracy += curr_accuracy 

	print("Testing accuracy: " + str(accuracy / len(accuracy_per_fold)))
	print("Accuracy per fold: " + str(accuracy_per_fold))
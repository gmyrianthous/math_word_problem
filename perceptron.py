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

# MultiArith: 6-fold cross validation
# 	100 addition followed by subtraction
# 	100 subtraction followed by addition
# 	100 addition and multiplication
# 	100 addition and division
# 	100 subtraction and multiplication
# 	100 subtraction and division.
def k_fold_cross_validation(data, dataset = "MultiArith"):
	# tuple(training, testing)
	k_folds = []
	if dataset == "MultiArith":
		k_folds.append((data[100:], data[:100]))
		k_folds.append((data[:100] + data[200:], data[100:200]))
		k_folds.append((data[:200] + data[300:], data[200:300]))
		k_folds.append((data[:300] + data[400:], data[300:400]))
		k_folds.append((data[:400] + data[500:], data[400:500]))
		k_folds.append((data[:500], data[500:]))

	return k_folds


def get_numbers_in_template(alignment, question):
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

# Get the words between two parameters
# start -> starting index
# end -> ending index
def get_words_between(start, end, question):
	if start > end:
		tmp = start
		start = end
		end = tmp

	words_between = question[start:end].split(" ")
	words_between.pop(0) # remove the first instance which is the first parameter
	words_between.pop(len(words_between)-1) # remove the last instance which is always empty


	#TODO

	#processed_words = []
	#for word in words_between:
		# Treat numbers between two numbers
		# NUM is a general placeholder
		#if word.isdigit():
		#	processed_words.append("NUM")
		#else:
			# Remove special characters
	#	processed_word = re.sub('[^A-Za-z0-9]+', '', word)
	#	processed_words.append(processed_word)


	# Discard nouns
	# NOTE: nltk.pos_tag accepts a LIST of tokens
	final_words = []
	#pos_tags = pos_tagger.tag(processed_words)
	#for pos_tag in pos_tags:
	#	if not 'NN' in pos_tag[1]:
	#		final_words.append(pos_tag[0])
	#for word in processed_words:
	#	word_in_list = []
	#	word_in_list.append(word)

	#	pos_tag = pos_tagger.tag(word_in_list)[0][1]
	#	if not 'NN' in pos_tag:
	#		final_words.append(word)

	return final_words

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

# Transform a op (b op c) to (a op b) op c
def convert_template(equation, alignment, numbers, operations):

	numbers_new = [numbers[1], numbers[2], numbers[0]]
	alignment_new = [alignment[1], alignment[2], alignment[0]]
	operations_new = operations[::-1]
	equation_new = "X=(("+str(float(numbers_new[0]))+operations_new[0]+str(float(numbers_new[1]))+")"+operations_new[1]+str(float(numbers_new[2]))+")"

	return equation_new, alignment_new, numbers_new, operations_new

# Function to compute the dot product between a combination's feature list and the weight vector
def dot_product(features_dict, weights):
	dot = 0
	for feature in features_dict.keys():
		if feature in weights.keys():
			dot += features_dict[feature] * weights[feature]
	return dot

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
		if "))" in equation:
			#print("modified")
			equation, alignment, numbers, operations = convert_template(equation, alignment, numbers, operations)


		# Feature 1: words between two numbers along with the operation. e.g. and_+, gave_- etc. 

		## Words between the first two parameters a and b
		#words_between = get_words_between(alignment[0], alignment[1], question)

		#for word in words_between:
		#	features[word.lower()+"_"+operations[0]] += 1

		## Words between the remaining two parameters
		#words_between = get_words_between(alignment[1], alignment[2], question)

		#for word in words_between:
		#	features[word.lower()+"_"+operations[1]] += 1


		# Feature 2: previousWord:word_operation_-
		# Get previous word
		previous_words = []
		for i in range(0, len(alignment)):
			index = alignment[i]-2
			reversed_word = ""
			curr_char = question[index]
			while curr_char != " ":
				reversed_word += curr_char
				index -= 1
				curr_char = question[index] 

			previous_words.append(reversed_word[::-1])

		# For the first number that appears in the template the sign is always positive
		features['previousWord:'+str(previous_words[0])+"_operation:+"] += 1
		features['previousWord:'+str(previous_words[1])+"_operation:"+operations[0]] += 1
		features['previousWord:'+str(previous_words[2])+"_operation:"+operations[1]] += 1


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


	return id_features_dict

# Function that produces all the possible combinations of the template
# Total: 96 operations
def get_equation_combinations():
	symbol = ['a', 'b', 'c']
	op = ['+', '-', '/', '*']
	combinations = []

	for symbols in itertools.permutations(symbol):
	    for ops in itertools.product(op, repeat=2):
	        combinations.append("(%s %s %s) %s %s" % (
	            symbols[0], ops[0], symbols[1], ops[1], symbols[2]))

	return combinations

# Extract combination features
def extract_combination_features(problem, combination):
	features = Counter()

	alignment = problem['alignment']
	numbers = problem['numbers']
	question = problem['question']

	# Feature 1: words between two numbers along with the operation. e.g. and_+, gave_- etc. 
	# Get the operations of the combination
	operations = get_operations_in_template(combination)


	previous_words = problem['previousWords']

	# For the first number that appears in the template the sign is always positive
	features['previousWord:'+str(previous_words[0])+"_operation:+"] += 1
	features['previousWord:'+str(previous_words[1])+"_operation:"+operations[0]] += 1
	features['previousWord:'+str(previous_words[2])+"_operation:"+operations[1]] += 1

	#print(question)
	#print(combination)
	#print(operations)
	#print(features)
	#print(previous_words)
	#print("")

	#sys.exit(1)


	## Words between the first two parameters a and b
	#words_between = get_words_between(alignment[0], alignment[1], question)

	#for word in words_between:
	#	features[word.lower()+"_"+operations[0]] += 1

	## Words between the remaining two parameters
	#words_between = get_words_between(alignment[1], alignment[2], question)

	#for word in words_between:
	#	features[word.lower()+"_"+operations[1]] += 1

	return features	


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
			elif char == 'c':
				combination_filled_in += str(numbers[2])
			else:
				combination_filled_in += char	

		# Extract features for the current combination
		combination_features = extract_combination_features(problem, combination_filled_in)

		combination_result = eval(combination_filled_in)
		#if 'how many' in question.lower():
		#	if isinstance(combination_result, int) and combination_result > 0:
		#		dot = dot_product(combination_features, weights)
		#		if (max_dot is None or dot > max_dot):
		#			max_dot = dot
		#			max_features = combination_features
		#			max_combination = combination_filled_in
		#elif 'how much' in question.lower():
		#	if isinstance(combination_result, float) and combination_result > 0:
		#		dot = dot_product(combination_features, weights)
		#		if (max_dot is None or dot > max_dot):
		#			max_dot = dot
		#			max_features = combination_features
		#			max_combination = combination_filled_in
		#else:
		dot = dot_product(combination_features, weights)
		if (max_dot is None or dot > max_dot):
			max_dot = dot
			max_features = combination_features
			max_combination = combination_filled_in

	return max_dot, max_features, max_combination

# Train the strucutred perceptron and learn the weights. 
# Multiple passes, shuffline and averaging can improve the performance of our classifier. 
def train(data, iterations=10):
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
def test(data, weights):
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
	# PoS tagger
	pos_tagger = PerceptronTagger()
	
	# Fix random seed so that results are reproducible. 
	random.seed(26)

	# Read the data
	data = read_data_json('data/MultiArith.json')

	# Report dataset's stats -> stdout
	report_dataset_stats(data)

	# Perform 6-fold cross validation -> list of 6 tuples(training, testing)
	six_folds = k_fold_cross_validation(data)

	accuracy_per_fold = []
	accuracy = 0 
	for i in range(0, len(six_folds)):
		print("Fold "+ str(i+1) + "/"+str(len(six_folds)))
		training_data = six_folds[i][0]
		testing_data = six_folds[i][1]

		# Extract features for training and testing examples
		training_features = extract_features(training_data)
		testing_features = extract_features(testing_data)

		# Train the structured perceptron in order to learn the weights
		feature_weights = train(training_features)

	#	print("")
	#	print(feature_weights)
	#	print("")

		# Test the perceptron
		curr_accuracy = test(testing_features, feature_weights)
		accuracy_per_fold.append(curr_accuracy)
		accuracy += curr_accuracy 

	print("Testing accuracy: " + str(accuracy / len(accuracy_per_fold)))
	print("Accuracy per fold: " + str(accuracy_per_fold))
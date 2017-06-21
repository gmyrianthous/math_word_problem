import json 
import re
import string
from collections import OrderedDict, Counter
import nltk
import random 
import itertools
import sys 
from random import shuffle

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


def cross_validation(data):
	return data[:500], data[500:]


# Extract features for every problem/question in the dataset. 
# Features: num1, num2, num3, words before/after the numerical indices (window size: 3)
#           PoS tag of words before/after the numerical indices
def extract_features(data):
	counter = 0 
	for problem in data:
		print(counter)
		#if counter == 0:
		question = problem['sQuestion']
		alignment = problem['lAlignments']
		print("Question: " + question)
		print("Alignment: " + str(alignment))

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

		# Extract words before/after the numerical indices. 
		# 1. Previous words n-1
		previous_words = []
		for index in alignment: 
			index_reversed = index-2 # index-1 = whitespace
			previous_word_reversed = ""
			while not question[index_reversed].isspace():
				previous_word_reversed += question[index_reversed]
				index_reversed -= 1
			previous_words.append(previous_word_reversed[::-1])
		print("Previous n-1 word: " + str(previous_words))

		# 2. Previous words n-2
		previous_words_2 = []
		for i in range(0, len(alignment)):
			index_reversed = alignment[i]-2-len(previous_words[i])-1
			if index_reversed > 0:
				previous_2nd_word_reversed = ""
				while not question[index_reversed].isspace():
					previous_2nd_word_reversed += question[index_reversed]
					index_reversed -= 1
				previous_words_2.append(previous_2nd_word_reversed[::-1])
			else:
				# Not found
				previous_words_2.appned("not_found")
		print("Previous n-2 word:" + str(previous_words_2))

		# 3. Next n+1 words
		next_words = []
		for i in range(0, len(alignment)):
			next_word_index = alignment[i]+len(str(numbers[i]))+1 # index+1 = whitespace
			next_word = ""
			if question[alignment[i] + len(str(numbers[i]))] in string.punctuation: # No next word was found
				next_words.append("not_found")
			else:
				while not question[next_word_index].isspace():
					next_word += question[next_word_index]
					next_word_index += 1
				next_words.append(next_word)
		print("Next n+1 word:" + str(next_words))

		# 4. Next n+2 words
		next_words_2 = []
		for i in range(0, len(alignment)):
			next_2nd_word = ""
			# If n+1 word was not found, then n+2 cannot be found. 
			if next_words[i] == "not_found" or next_words[i] == "":
				next_words_2.append("not_found")
			else:
				next_2nd_word_index = alignment[i] + len(str(numbers[i])) + len(next_words[i]) + 2
				if next_2nd_word_index >= len(question):
					next_words_2.append("not_found")
				else:
					while not question[next_2nd_word_index].isspace():
						next_2nd_word += question[next_2nd_word_index]
						next_2nd_word_index += 1
					next_words_2.append(next_2nd_word)
		print("Next n+2 word: " + str(next_words_2))


		counter+=1

def extract_features2(data):
	id_features_dict = OrderedDict() # sentence id (key) : list(features)
	counter = 0

	removed_instances = []

	for math_problem in data:
		if counter >= 0 and counter not in removed_instances:
			question = math_problem['sQuestion']
			alignment = math_problem['lAlignments']
			equation = str(math_problem['lEquations'][0])
			index = math_problem['iIndex']
			features = Counter()

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

			# ***NOTE***: Numbers extracted appear in the right sequence that should be used to fill-in the template. 
			#print("Numbers extracted from alignments: " + str(numbers))

			# Feature 1: number_parameter features (e.g. num1_a, num2_c, num3_b)
			original_number_order = sorted(alignment, reverse=False) # order in which the numbers appear in the question (not in the template)

			for i in range(0, len(original_number_order)):
				if alignment.index(original_number_order[i]) == 0:
					features["num"+str(i+1)+"_a"] += 1
				elif alignment.index(original_number_order[i]) == 1:
					features["num"+str(i+1)+"_b"] += 1
				else:
					features["num"+str(i+1)+"_c"] += 1

			# Compute the numbers as appear in the text (not in template)
			num1 = -1
			num2 = -1
			num3 = -1

			if alignment[0] < alignment[1] < alignment[2]: # num1, num2, num3
				num1 = numbers[0]
				num2 = numbers[1]
				num3 = numbers[2]
			elif alignment[0] < alignment[2] < alignment[1]: 
				num1 = numbers[0]
				num2 = numbers[2]
				num3 = numbers[1]
			elif alignment[1] < alignment[0] < alignment[2]:
				num1 = numbers[1]
				num2 = numbers[0]
				num3 = numbers[2]
			elif alignment[1] < alignment[2] < alignment[0]: 
				num1 = numbers[1]
				num2 = numbers[2]
				num3 = numbers[0]
			elif alignment[2] < alignment[0] < alignment[1]:
				num1 = numbers[2]
				num2 = numbers[0]
				num3 = numbers[1]
			else:
				num1 = numbers[2]
				num2 = numbers[1]
				num3 = numbers[0]

			# Feature 2: parameter1_verb_parameter2_verb->op#_'op' (e.g. a_had_b_had->op1_+)



			# Get the sentences of each problem/question
			sentences = re.split(r'[.!?]+', question)
			sentences.pop(len(sentences)-1) # delete the last entry in the list which is always empty

			# Discard sentences without any numerical instances.
			sentence_contains_numbers = []
			for sentence in sentences:
				sentence = re.sub('[^a-zA-Z0-9 \n\.]', '', sentence) # deal with dollar sign issue
				contains_nums = False
				words = sentence.split(" ")
				for word in words:
					if word.isdigit() or '$' in word:
						contains_nums = True
				sentence_contains_numbers.append(contains_nums)

			sentences_final = []
			for i in range(0, len(sentences)):
				if sentence_contains_numbers[i]:
					sentences_final.append(sentences[i])

			#sentences_removed = 0
			#for i in range(0, len(sentences)):
			#	if not sentence_contains_numbers[i-sentences_removed]:
			#		sentences.pop(i-sentences_removed)
					#sentences_removed+=1
			#print(sentences_final)

			# We need to compute the PoS tags in order to get the verbs which are related to the parameters a, b and c.
			nums_with_verbs = []
			for sentence in sentences_final:
				#print(sentence)
				sentence_pos = nltk.pos_tag(nltk.word_tokenize(sentence))
				#print(sentence_pos)
				for i in range(0, len(sentence_pos)):
					if sentence_pos[i][0].isdigit() and 'CD' == sentence_pos[i][1]:
						if int(sentence_pos[i][0]) in numbers:
							# Find the relevant verb (i.e. the nearest) starting from left... 
							verb_found = False
							for j in range(i, 0, -1):
								if 'VB' in sentence_pos[j][1]:
									nums_with_verbs.append((sentence_pos[i][0], sentence_pos[j][0]))
									verb_found = True
									break

							# ... to right.
							if not verb_found:
								for j in range(len(sentence_pos)-1, i, -1):
									if 'VB' in sentence_pos[j][1]:
										nums_with_verbs.append((sentence_pos[i][0], sentence_pos[j][0]))
										verb_found = True
										break

							# Extreme scenario -> something wrong with the dataset entries. 
							if not verb_found:
								nums_with_verbs.append((sentence_pos[i][0], "NOT_FOUND"))
		
			#print(nums_with_verbs)

			# Now we need to sort the list according to the sequence that numbers will appear in the template. 
			nums_with_verbs_sorted = []

			alignment_a = alignment[0]
			alignment_b = alignment[1]
			alignment_c = alignment[2]

			if (alignment_a > alignment_b) and (alignment_b > alignment_c): # a b c
				nums_with_verbs_sorted = nums_with_verbs
			elif (alignment_a > alignment_c) and (alignment_c > alignment_b): # a c b
				nums_with_verbs_sorted.append(nums_with_verbs[0])
				nums_with_verbs_sorted.append(nums_with_verbs[2])
				nums_with_verbs_sorted.append(nums_with_verbs[1])
			elif (alignment_c > alignment_a) and (alignment_a > alignment_b): # c a b
				nums_with_verbs_sorted.append(nums_with_verbs[2])
				nums_with_verbs_sorted.append(nums_with_verbs[0])
				nums_with_verbs_sorted.append(nums_with_verbs[1])
			elif (alignment_c > alignment_b) and (alignment_b > alignment_a): # c b a
				nums_with_verbs_sorted.append(nums_with_verbs[2])
				nums_with_verbs_sorted.append(nums_with_verbs[1])
				nums_with_verbs_sorted.append(nums_with_verbs[0])
			elif (alignment_b > alignment_a) and (alignment_a > alignment_c): # b a c
				nums_with_verbs_sorted.append(nums_with_verbs[1])
				nums_with_verbs_sorted.append(nums_with_verbs[0])
				nums_with_verbs_sorted.append(nums_with_verbs[2])
			elif (alignment_b > alignment_c) and (alignment_c > alignment_a): # b c a
				nums_with_verbs_sorted.append(nums_with_verbs[1])
				nums_with_verbs_sorted.append(nums_with_verbs[2])
				nums_with_verbs_sorted.append(nums_with_verbs[0])	
			else:
				print("ERROR: Failed to sort nums_with_verbs list.")

			nums_with_verbs_sorted = nums_with_verbs_sorted[::-1]	# index=0 -> a, index=1 -> b, index=2 -> c	
			#print("Nums with verbs - sorted: " + str(nums_with_verbs_sorted))

			# Extract the signs/operations from the equation
			# form left to right
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


			# Construct the final features 
			# a op b
			features["a_"+nums_with_verbs_sorted[0][1]+"_b_"+nums_with_verbs_sorted[1][1]+"->op1_"+operations[0]] += 1
			# b op c
			features["b_"+nums_with_verbs_sorted[0][1]+"_c_"+nums_with_verbs_sorted[1][1]+"->op2_"+operations[1]] += 1

			id_features_dict[counter] = features

			id_features_dict[counter] = {}
			id_features_dict[counter]['features'] = features
			id_features_dict[counter]['question'] = question
			id_features_dict[counter]['equation'] = equation
			id_features_dict[counter]['numbers'] = numbers # in the correct sequence that matches the tempalte
			id_features_dict[counter]['num1'] = num1 # first number that appears in the text
			id_features_dict[counter]['num2'] = num2 # second number that appears in the text
			id_features_dict[counter]['num3'] = num3 # third number that appears in the text
			id_features_dict[counter]['alignment'] = alignment
			id_features_dict[counter]['nums_with_verbs_sorted'] = nums_with_verbs_sorted
			id_features_dict[counter]['solution'] = math_problem['lSolutions']

		counter += 1

	return id_features_dict

# Initialise a dictionary of weights. 
def initialise_weights(featurised_data):
	weights = OrderedDict()

	for sentence in featurised_data.values():
		for feature in sentence['features']:
			if feature not in weights.keys():
				weights[feature] = 0
	return weights

# Function for shuffling an OrderedDict
def shuffle_data(data):
    # Shuffle the keys
    keys = list(data.keys())
    # Fix the random seed so that results are reproducible. 
    random.shuffle(keys)

    # Write the shuffled keys in another dictionary.
    shuffled_dictionary = OrderedDict()
    for key in keys:
        shuffled_dictionary[key] = data[key]

    return shuffled_dictionary


# Function for computing all the combinations of the equation (a op b) op c
def get_equation_combinations():
	symbol = ['a', 'b', 'c']
	op = ['+', '-', '/', '*']
	combinations = []
	for symbols in itertools.permutations(symbol):
	    for ops in itertools.product(op, repeat=2):
	        combinations.append("(%s %s %s) %s %s" % (
	            symbols[0], ops[0], symbols[1], ops[1], symbols[2]))
	return combinations


# Extract combination's features
def extract_combination_features(problem, combination):
	alignment = problem['alignment']
	#numbers = problem['numbers'] # in the correct sequence wrt the template
	num1 = problem['num1']
	num2 = problem['num2']
	num3 = problem['num3']
	features = Counter()

	# Feature 1: number_parameter features (e.g. num1_a, num2_c, num3_b)
	# Extract the numbers as appear in this combination
	numbers = []
	temp = re.sub('[()]', '', combination)
	temp2 = temp.split(" ")
	for t in temp2:
		if t.isdigit():
			numbers.append(int(t))

	num1_found = False
	num2_found = False
	num3_found = False

	parameter = {}
	parameter[0] = 'a'
	parameter[1] = 'b'
	parameter[2] = 'c'

	for i in range(0, len(numbers)): # go through the numbers in the sequence that appear in the template
		if numbers[i] == num1 and not num1_found:
			features['num1_'+parameter[i]] += 1
			num1_found = True
		elif numbers[i] == num2 and not num2_found:
			features['num2_'+parameter[i]] += 1 
			num2_found = True
		elif numbers[i] == num3 and not num3_found:
			features['num3_'+parameter[i]] += 1
			num3_found = True
		else: 
			print("ERROR: Failed to extract number_parameter feature for combination!")


	#print(combination)
	#print(alignment)
	#print(numbers)
	#print("Num1: " + str(num1) + ", Num2: " + str(num2) + ", Num3: " + str(num3))

	# Feature 2: parameter1_verb_parameter2_verb->op#_'op' (e.g. a_had_b_had->op1_+)
	nums_with_verbs = problem['nums_with_verbs_sorted']
	#print(nums_with_verbs)

	# Sort nums_with_verbs according to the new template. 
	nums_with_verbs_sorted = []


	for i in range(0, len(numbers)):
		for j in range(0, len(nums_with_verbs)):
			if numbers[i] == int(nums_with_verbs[j][0]):
				nums_with_verbs_sorted.append(nums_with_verbs[j])
	#print(nums_with_verbs_sorted)

	# Construct features
	# Extract the signs/operations from the equation
	# form left to right
	operations = []
	for char in combination:
		if char == "+":
			operations.append("+")
		elif char == "-":
			operations.append("-")
		elif char == "*":
			operations.append("*")
		elif char == "/":
			operations.append("/")


	# Construct the final features 
	# a op b
	features["a_"+nums_with_verbs_sorted[0][1]+"_b_"+nums_with_verbs_sorted[1][1]+"->op1_"+operations[0]] += 1
	# b op c
	features["b_"+nums_with_verbs_sorted[0][1]+"_c_"+nums_with_verbs_sorted[1][1]+"->op2_"+operations[1]] += 1	

	#print("Features: " + str(features))
	#print("")

	return features



# Function to compute the dot product between a combination's feature list and the weight vector
def dot_product(features_dict, weights):
	dot = 0
	for feature in features_dict.keys():
		if feature in weights.keys():
			dot += features_dict[feature] * weights[feature]
	return dot


# Function to compute argmax operation. 
# TEMPLATE: (a op b) op c
def argmax(problem, weights):
	question = problem['question']
	features = problem['features']
	numbers = problem['numbers'] # sequence: a, b, c
	nums_with_verbs_sorted = problem['nums_with_verbs_sorted'] # index 0 -> a, index 1 -> b, index 2 -> c
 

	# Compute the possible combinations for the problem. 
	combinations = get_equation_combinations()


	max_dot = None 
	max_features = None
	max_combination = None

	# Construct each combination and compute the features and score for each. 
	for combination in combinations:
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

		# Extract features for the combination. 
		combination_features = extract_combination_features(problem, combination_filled_in)

		# Compute the score of the combination
		dot = dot_product(combination_features, weights)

		if (max_dot is None or dot > max_dot):
			max_dot = dot
			max_features = combination_features
			max_combination = combination_filled_in

	return max_dot, max_features, max_combination



# Training a structured perceptron to learn the features' weights. 
def train(data, iterations=10):
	# Initialize weights 
	weights = initialise_weights(data)

	for i in range(iterations):
		# Shuffle data
		data = shuffle_data(data)
		counter = 0
		for problem in data:
			if counter >= 0:
				features = data[problem]['features']

				# Predict the sequence of numbers and operations according to the template. 
				y_hat_dot, y_hat_features, y_hat_combination = argmax(data[problem], weights)


				#print(y_hat_features)
				correct_features = []
				wrong_features = []

				# Compare the prediction with the true tag sequence
				for feature in features:
					if feature not in y_hat_features:
						correct_features.append(feature)

				for feature in y_hat_features:
					if feature not in features.keys():
						wrong_features.append(feature)

				# Update weights according to the comparison above
				for feature in correct_features:
					if feature in weights.keys():
						weights[feature] += 1

				for feature in wrong_features:
					if feature in weights.keys():
						weights[feature] -= 1

			counter += 1

	return weights

# Function used for testing the structured perceptron
def test(data, weights):
	correct_predictions = 0
	counter=0
	for problem in data:
		features = data[problem]['features']
		solution = data[problem]['solution']
	
		y_hat_dot, y_hat_features, y_hat_combination = argmax(data[problem], weights)

		print(data[problem]['question'])
		print(solution)
		print(y_hat_combination)

		# Execute the predicted equation
		prediction = eval(y_hat_combination)

		if int(prediction) == int(solution[0]):
			correct_predictions += 1

	print("Correct predictions: " + str(correct_predictions))
	print("Accuracy: " + str(correct_predictions / len(data) * 100))





if __name__ == "__main__":
	random.seed(26)

	# Read the data
	data = read_data_json("data/MultiArith.json")
	shuffle(data)	
	#pprint(data)

	# Report dataset stats
	report_dataset_stats(data)

	# Split the data into testing and training sents
	training_data, testing_data = cross_validation(data)

	# Extract features of training and testing data points
	featurised_sentences_dict = extract_features2(training_data) 
	featurised_sentences_testing_dict = extract_features2(testing_data)

	# Train a structured perceptron to learn the weights
	feature_weights = train(featurised_sentences_dict)

	test(featurised_sentences_testing_dict, feature_weights)




import json 
import re
import string
from collections import OrderedDict, Counter
import nltk

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

			print("Question: " + question)
			print("Equation: " + equation)
			print("Alignment:" + str(alignment))

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
			print("Numbers extracted from alignments: " + str(numbers))

			# Feature 1: number_parameter features (e.g. num1_a, num2_c, num3_b)
			original_number_order = sorted(alignment, reverse=False) # order in which the numbers appear in the question (not in the template)

			for i in range(0, len(original_number_order)):
				if alignment.index(original_number_order[i]) == 0:
					features["num"+str(i+1)+"_a"] += 1
				elif alignment.index(original_number_order[i]) == 1:
					features["num"+str(i+1)+"_b"] += 1
				else:
					features["num"+str(i+1)+"_c"] += 1


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
			print(sentences_final)

			# We need to compute the PoS tags in order to get the verbs which are related to the parameters a, b and c.
			nums_with_verbs = []
			for sentence in sentences_final:
				print(sentence)
				sentence_pos = nltk.pos_tag(nltk.word_tokenize(sentence))
				print(sentence_pos)
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
		
			print(nums_with_verbs)

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
			print("Nums with verbs - sorted: " + str(nums_with_verbs_sorted))

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

		counter += 1






if __name__ == "__main__":
	# Read the data
	data = read_data_json("data/MultiArith.json")
	#pprint(data)

	# Report dataset stats
	report_dataset_stats(data)

	# Split the data into testing and training sents
	training_data, testing_data = cross_validation(data)

	# Extract features
	#extract_features(data)

	# Extract features of training data points
	featurised_sentences_dict = extract_features2(training_data) 


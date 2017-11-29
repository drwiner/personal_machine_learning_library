# from SVM import svm
from collections import namedtuple, defaultdict
import numpy as np
import random

ZERO_PAD = 300

LabeledEx = namedtuple('LabeledEx', ['label', 'feats'])


# def dot_w_indices(feat_indices, np_array):
# 	return sum(np_array[i] for i in feat_indices)
#
#
# def get_random_example(examples):
# 	rand_int = random.randrange(0, len(examples))
# 	return examples[rand_int]

def logreg(examples, weights, bias, tradeoff, learn_rate, epochs):
	for epoch in range(epochs):
		pass
		# example = get_random_example(examples)


def svm(examples, weights, bias, tradeoff, learn_rate, epochs):
	pass
# def svm(examples, weights, bias, tradeoff, learn_rate, epochs):
# 	# num_errors = 0
# 	for epoch in range(epochs):
#
# 		for example in examples:
# 			if example.label * dot_w_indices(example.feats, np.transpose(weights)) <= 1:
# 				weights = (1 - learn_rate) * weights + learn_rate * tradeoff * example.label * example.feats
#
# 		# Randomly pick an example
# 		example = get_random_example(examples)
#
# 		# Treat example as a full data set, take derivative of objective at current weight
# 		margin = 0.5 * np.transpose(weights) * weights
# 		slack = max(0, 1 - example.label * dot_w_dict(example.feats, np.transpose(weights)))
# 		deriv = margin + tradeoff * slack
#
# 		# update
# 		weights = weights - learn_rate * deriv
# 		bias = bias + learn_rate*example.label
#
# 	# print('updates:\t{}'.format(num_errors))
# 	return weights, bias


def run_svms(examples, largest_index):
	initial_weights = np.array([0.00001 for j in range(largest_index+3)])
	bias = 0.00001
	learn_rates = [10, 1, .1, .01, .001, .0001]
	tradeoffs = [10, 1, .1, .01, .001, .0001]

	results = []
	for lr in learn_rates:
		for to in tradeoffs:
			w, b = svm(examples, initial_weights, to, lr, 40)
			results.append((w,b))


def get_largest_index(training_whole):
	largest = 0
	with open(training_whole, 'r') as tw:
		for line in tw:
			linsp = line.split()
			cndt = int(linsp[-1].split(":")[0])
			if cndt > largest:
				largest = cndt
	return largest


def parse(file_name, num_feats):
	examples = []
	with open(file_name, 'r') as fn:
		for line in fn:
			linsp = line.split()
			label = int(linsp[0])
			feats = [feat.split(":")[0] for feat in linsp[1:]]
			feat_vec = np.zeros(num_feats + ZERO_PAD)
			for f in feats:
				feat_vec[f] = 1.0
			examples.append(LabeledEx(label, feat_vec))

	return examples


if __name__ == '__main__':
	# read input
	base_cvsplits = "data/CVSplits/training0{}.data"
	training_whole = "data/speeches.train.liblinear"
	test = "data/speeches.test.liblinear"

	# // largest index
	lfi = max(get_largest_index(training_whole), get_largest_index(test))

	training_examples = [parse(base_cvsplits.format(i), lfi) for i in range(5)]

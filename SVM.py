import numpy as np
import random
from collections import namedtuple, defaultdict
from clockdeco import clock
LabeledEx = namedtuple('LabeledEx', ['label', 'feats'])


# @clock
def svm(examples, weights, tradeoff, learn_rate, epochs):
	example_list = list(examples)
	initial_learning_rate = learn_rate
	bias = 0.0001


	for epoch in range(epochs):
		num_errors = 0
		for t, example in enumerate(example_list):
			feats_with_bias_term = np.append(example.feats, [1.0])
			p = example.label * np.dot(np.transpose(weights), feats_with_bias_term)
			q = weights - learn_rate
			if p <= 1:
				weights = q + learn_rate * tradeoff * example.label * feats_with_bias_term
				num_errors += 1
			else:
				weights = q

			bias = bias + learn_rate * example.label
			weights[-1] = bias
			learn_rate = initial_learning_rate / (1 + t)


		random.shuffle(example_list)
	# print("lr={}, tradeoff={}, acc={}\n".format(initial_learning_rate, tradeoff, num_errors / len(example_list)))
	print('learn_rate\t{}\ttradeoff\t{}\tacc\t{}'.format(initial_learning_rate, tradeoff, str(num_errors/len(example_list))))
	return weights


def test_svm(examples, weights):
	num_errors = 0
	for t, example in enumerate(examples):
		feats_with_bias_term = np.append(example.feats, [1.0])
		p = example.label * np.dot(np.transpose(weights), feats_with_bias_term)
		if p > 1:
			num_errors += 1

	print('TEST: acc\t{}'.format(str(1 - (num_errors / len(examples)))))


from ID3 import use_tree
from learning_util import get_largest_index


def transform_trees_to_feats(examples, trees):
	example_list = list(examples)
	for example in example_list:
		example.feats = np.array([use_tree(tree, example) for tree in trees])
	return example_list


if __name__ == '__main__':
	# read input

	test = "data/speeches.test.liblinear"
	training_whole = "data/speeches.train.liblinear"

	lfi = max(get_largest_index(training_whole), get_largest_index(test))

	from learning_util import parse

	examples = parse(training_whole, lfi, 0)

	test_examples = parse(test, lfi, 0)



	# dtrees = []
	# with open("dtrees.txt", 'r') as dtreefile:
	# 	for line in dtreefile:
	# 		dtrees.append(eval(line))
	#
	# transformed_examples = transform_trees_to_feats(examples, dtrees)
	# transformed_test_examples = transform_trees_to_feats(test_examples, dtrees)

import numpy as np
import random
from collections import namedtuple
import math
from ID3 import use_tree

LabeledEx = namedtuple('LabeledEx', ['label', 'feats'])


def get_random_example(examples):
	rand_int = random.randrange(0, len(examples))
	return examples[rand_int]


def exp(val):
	return math.e**val


def log_regression(examples, weights, sigma_squared, learn_rate, epochs):
	initial_learning_rate = learn_rate
	for epoch in range(epochs):
		example = get_random_example(examples)
		feats_with_bias_term = np.append(example.feats, [1.0])
		p = 1/(1 + exp(example.label * np.dot(np.transpose(weights), feats_with_bias_term)))
		tradeoff = (-4 * weights) / (math.sqrt(sigma_squared) ** 3)
		q = weights - learn_rate
		weights = q + p * learn_rate * tradeoff
		learn_rate = initial_learning_rate / (1 + epoch)
	return weights


def test_log_regression(examples, weights):
	num_errors = 0
	for example in examples:
		feats_with_bias_term = np.append(example.feats, [1.0])
		# p = 1/(1 + exp(example.label * np.dot(np.transpose(weights), feats_with_bias_term)))
		p = example.label * np.dot(np.transpose(weights), feats_with_bias_term)
		if p > 1:
			num_errors += 1

	print('TEST: acc\t{}'.format(str(1 - (num_errors / len(examples)))))


def transform_trees_to_feats(examples, trees):
	example_list = list(examples)
	for example in example_list:
		example.feats = np.array([use_tree(tree, example) for tree in trees])
	return example_list

from learning_util import get_largest_index, parse

if __name__ == '__main__':
	test = "data/speeches.test.liblinear"
	training_whole = "data/speeches.train.liblinear"

	lfi = max(get_largest_index(training_whole), get_largest_index(test))

	examples = parse(training_whole, lfi, 0)
	test_examples = parse(test, lfi, 0)

	dtrees = []
	with open("dtrees.txt", 'r') as dtreefile:
		for line in dtreefile:
			dtrees.append(eval(line))

	transformed_examples = transform_trees_to_feats(examples, dtrees)
	transformed_test_examples = transform_trees_to_feats(test_examples, dtrees)
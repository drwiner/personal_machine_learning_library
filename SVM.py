import numpy as np
import random
from collections import namedtuple

LabeledEx = namedtuple('LabeledEx', ['label', 'feats'])


def svm(examples, weights, tradeoff, learn_rate, epochs):
	example_list = list(examples)
	initial_learning_rate = learn_rate
	# num_errors = 0
	for epoch in range(epochs):

		for t, example in enumerate(example_list):
			p = example.label * np.transpose(weights) * example.feats
			q = weights - learn_rate
			if p <= 1:
				weights = q + learn_rate * tradeoff * example.label * example.feats
			else:
				weights = q

			learn_rate = initial_learning_rate / (1 + t)

		example_list = random.shuffle(example_list)

	return weights


from ID3 import use_tree

def transform_trees_to_feats(examples, trees):
	example_list = list(examples)
	for example in example_list:
		example.feats = np.array([use_tree(tree, example) for tree in trees])
	return example_list


if __name__ == '__main__':
	# read input
	base_cvsplits = "data/CVSplits/training0{}.data"
	training_whole = "data/speeches.train.liblinear"

	test = "data/speeches.test.liblinear"

	# // largest index
	lfi = max(get_largest_index(training_whole), get_largest_index(test))
	examples = parse(training_whole, lfi)

	# training_examples = [parse(base_cvsplits.format(i), lfi) for i in range(5)]
	for i in range(2):
		sub_set = get_x(examples, 100)
		dtree = ID3_depth(sub_set, range(lfi), 3)
		print(dtree)

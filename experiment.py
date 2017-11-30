# from SVM import svm
from collections import namedtuple, defaultdict
import numpy as np
import random
from SVM import svm, test_svm
from LOG_REGRESSION import log_regression, test_log_regression
from clockdeco import clock

ZERO_PAD = 300

LabeledEx = namedtuple('LabeledEx', ['label', 'feats'])

# @clock
def run_svms(example_sets, num_sets, largest_index):

	# each 'i' is test
	for i in range(len(example_sets)):
		# each 'j' is training
		training = []
		for j in range(len(example_sets)):
			if i == j:
				continue
			training.extend(example_sets[j])

		initial_weights = np.array([0.00001 for k in range(largest_index)] + [1])

		learn_rates = [10, 1, .1, .01, .001, .0001]
		tradeoffs = [10, 1, .1, .01, .001, .0001]

		# train with 4/5
		# this weight_vals_list has a position for each learning rate
		print('fold:\t{}'.format(str(i)))
		for lr in learn_rates:
			for to in tradeoffs:

				w = svm(training, initial_weights, to, lr, 40)
				test_svm(example_sets[i], w)


def run_logregressions(example_sets, num_sets, largest_index):

	# each 'i' is test
	for i in range(len(example_sets)):
		# each 'j' is training
		training = []
		for j in range(len(example_sets)):
			if i == j:
				continue
			training.extend(example_sets[j])

		initial_weights = np.array([0.00001 for k in range(largest_index)] + [1])

		learn_rates = [1, .1, .01, .001, .0001, .00001]
		sigmas = [.1, 1, 10, 100, 1000, 10000]

		# train with 4/5
		# this weight_vals_list has a position for each learning rate
		print('fold:\t{}'.format(str(i)))
		for lr in learn_rates:
			for sigma in sigmas:
				print('learn_rate\t{}\tsigma_squared\t{}'.format(lr, sigma))
				w = log_regression(training, initial_weights, sigma, lr, 400)
				test_log_regression(example_sets[i], w)


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
			feats = [int(feat.split(":")[0]) for feat in linsp[1:]]
			feat_vec = np.zeros(num_feats)
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

	# run_svms(training_examples, len(training_examples), lfi)
	# initial_weights = np.array([0.00001 for k in range(lfi+1)] + [1])
	# w = svm(parse(training_whole, lfi+1), initial_weights, 10, 0.0001, 40)
	# test_svm(parse(test, lfi+1), w)

	# run_logregressions(training_examples, len(training_examples), lfi)


	initial_weights = np.array([0.00001 for k in range(lfi+1)] + [1])
	w = log_regression(parse(training_whole, lfi+1), initial_weights, 10, 0.0001, 40)
	test_log_regression(parse(test, lfi+1), w)
# from SVM import svm
from collections import namedtuple, defaultdict
import numpy as np
import random
from SVM import svm, test_svm
from LOG_REGRESSION import log_regression, test_log_regression
from NAIVE_BAYES import parse_featlist, naive_bayes, run_naive_bayes, test_naive_bayes
from ID3 import get_x_trees, use_tree
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
	lr_sigma_dict = defaultdict(lambda: {.1:0, 1:0, 10:0, 100:0, 1000:0, 10000:0})
	learn_rates = [1, .1, .01, .001, .0001, .00001]
	sigmas = [.1, 1, 10, 100, 1000, 10000]

	for i in range(len(example_sets)):
		# each 'j' is training
		training = []
		for j in range(len(example_sets)):
			if i == j:
				continue
			training.extend(example_sets[j])

		initial_weights = np.array([0.00001 for k in range(largest_index)] + [1])

		# train with 4/5
		# this weight_vals_list has a position for each learning rate
		print('fold:\t{}'.format(str(i)))
		for lr in learn_rates:
			for sigma in sigmas:
				print('learn_rate\t{}\tsigma_squared\t{}'.format(lr, sigma))
				w = log_regression(training, initial_weights, sigma, lr, 400)
				acc = test_log_regression(example_sets[i], w)
				lr_sigma_dict[lr][sigma] += acc

	print('CV averages:')
	for lr in learn_rates:
		for sigma in sigmas:
			print("LR:\t{}\tsigma^2:\t{}\tacc:\t{}".format(lr, sigma, lr_sigma_dict[lr][sigma] / num_sets))


def get_largest_index(training_whole):
	largest = 0
	with open(training_whole, 'r') as tw:
		for line in tw:
			linsp = line.split()
			cndt = int(linsp[-1].split(":")[0])
			if cndt > largest:
				largest = cndt
	return largest


def compile_trees(examples):
	dtrees = get_x_trees(examples, 100)

	with open("dtrees3.txt", 'w') as dtree_file:
		for dtree in dtrees:
			dtree_file.write(str(dtree))
			dtree_file.write('\n')


def run_bagged_forest_prediction(examples_bt, test_examples_bt):

	num_correct = 0
	print("accuracy for bagged trees: training")
	for i, example in enumerate(examples_bt):
		print(i)
		decision = 0
		for dtree in dtrees:
			decision += use_tree(dtree, example)
		if decision > 0 and example.label == 1:
			num_correct += 1
		elif decision <= 0 and example.label == -1:
			num_correct += 1

	print(num_correct / len(examples_bt))

	num_correct = 0
	print("accuracy for bagged trees: test")
	for example in test_examples_bt:
		decision = 0
		for dtree in dtrees:
			decision += use_tree(dtree, example)
		if decision > 0 and example.label == 1:
			num_correct += 1
		elif decision <= 0 and example.label == -1:
			num_correct += 1

	print(num_correct / len(test_examples_bt))


def transform_trees_to_feats(examples, trees):
	example_list = list(examples)
	new_list = []
	for example in example_list:
		feat_array = np.array([use_tree(tree, example) for tree in trees])
		new_list.append(LabeledEx(example.label, feat_array))
	return new_list


def run_svm_over_trees(_split_examples, _training_examples, _test_examples, dtrees, lfi):
	split_examples = []
	for tset in _split_examples:
		transformed_tset = transform_trees_to_feats(tset, dtrees)
		split_examples.append(transformed_tset)

	run_svms(split_examples, len(split_examples), lfi)

	# training_examples = transform_trees_to_feats(examples, dtrees)
	#
	# run_svms(training_examples, len(training_examples), lfi)
	# w = svm(examples, initial_weights, 10, 0.0001, 40)
	# test_svm(test_examples, w)


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

	do_log_regr = 0
	do_svm = 0
	do_naive_bayes = 0
	do_compile_trees = 0
	do_bagged_trees = 0
	do_svm_over_trees = 1

	# // largest index
	lfi = max(get_largest_index(training_whole), get_largest_index(test))

	training_examples = [parse(base_cvsplits.format(i), lfi) for i in range(5)]
	examples = parse(training_whole, lfi + 1)
	test_examples = parse(test, lfi + 1)
	initial_weights = np.array([0.00001 for k in range(lfi + 1)] + [1])

	if do_svm:
		run_svms(training_examples, len(training_examples), lfi)
		w = svm(examples, initial_weights, 10, 0.0001, 40)
		test_svm(test_examples, w)

	if do_log_regr:

		# run cv splits
		print("log regr cv splits")
		run_logregressions(training_examples, len(training_examples), lfi)

		# all training examples
		w = log_regression(examples, initial_weights, 10000, 0.0001, 40)

		# test on training
		acc = test_log_regression(examples, w)
		print("log regr train whole acc:\t{}".format(acc))

		# test on test
		acc = test_log_regression(test_examples, w)
		print("log regr test_acc:\t{}".format(acc))

	examples_nb = None
	test_examples_nb = None
	examples_bt = None
	test_examples_bt = None

	if do_naive_bayes:
		run_naive_bayes(training_examples, len(training_examples), lfi)

		examples_nb = parse_featlist(training_whole, lfi)
		p0, p1, pminus0, pminus1 = naive_bayes(examples_nb, 0.5, lfi)
		# test_naive_bayes(examples, lfi, p0, p1)

		test_examples_nb = parse_featlist(test, lfi)
		print('naive bayes on whole training')
		test_naive_bayes(examples_nb, lfi, p0, p1, pminus0, pminus1)

		print('naive bayes on test')
		test_naive_bayes(test_examples_nb, lfi, p0, p1, pminus0, pminus1)

	if do_compile_trees:
		dtrees = get_x_trees(examples, 100)

		with open("dtrees3.txt", 'w') as dtree_file:
			for dtree in dtrees:
				dtree_file.write(str(dtree))
				dtree_file.write('\n')

	if do_bagged_trees:

		dtrees = []
		with open("dtrees3.txt", 'r') as dtree_files:
			for line in dtree_files:
				dtree = eval(line)
				dtrees.append(dtree)

		if examples_nb is None:
			examples_bt = parse_featlist(training_whole, lfi)
		else:
			examples_bt = examples_nb
		if test_examples_nb is None:
			test_examples_bt = parse_featlist(test, lfi)
		else:
			test_examples_bt = test_examples_nb

		run_bagged_forest_prediction(examples_bt, test_examples_bt)


	if do_svm_over_trees:
		dtrees = []
		with open("dtrees3.txt", 'r') as dtree_files:
			for line in dtree_files:
				dtree = eval(line)
				dtrees.append(dtree)

		if examples_bt is None:
			if examples_nb is None:
				examples_bt = parse_featlist(training_whole, lfi)
			else:
				examples_bt = examples_nb

		if test_examples_bt is None:
			if test_examples_nb is None:
				test_examples_bt = parse_featlist(test, lfi)
			else:
				test_examples_bt = test_examples_nb

		split_examples = [parse_featlist(base_cvsplits.format(i), lfi) for i in range(5)]
		run_svm_over_trees(split_examples, examples_bt, test_examples_bt, dtrees, lfi)
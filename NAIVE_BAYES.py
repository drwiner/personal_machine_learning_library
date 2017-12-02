import numpy as np
import random
from collections import namedtuple, defaultdict
import math
from learning_util import get_largest_index
from clockdeco import clock

LabeledEx = namedtuple('LabeledEx', ['label', 'feats'])


def likelihood(count_feat_label, gamma, count_label):
	return (count_feat_label + gamma) / (count_label + 2*gamma)


def naive_bayes(examples, gamma, lfi):
	print('learning naive bayes')
	dd_0 = defaultdict(int)
	dd_1 = defaultdict(int)
	num_pos = 0
	for example in examples:

		for ef in example.feats:

			if example.label == 1:
				dd_1[ef] += 1

			else:
				dd_0[ef] += 1
		if example.label == 1:
			num_pos += 1


	prob_1_per_feat = [0]
	priorminus_1_per_feat = [0]
	prob_0_per_feat = [0]
	priorminus_0_per_feat = [0]

	prior0 = ((len(examples) - num_pos) / len(examples))

	prior1 = (num_pos / len(examples))


	for i in range(1, lfi+1):
		# print(i)
		p0 = likelihood(dd_0[i], gamma, len(examples)-num_pos)
		prob_0_per_feat.append(math.log(p0) + math.log(prior0))
		priorminus_0_per_feat.append(math.log(1-p0) + math.log(prior0))


		p1 = likelihood(dd_1[i], gamma, num_pos)
		prob_1_per_feat.append(math.log(p1) + math.log(prior1))
		priorminus_1_per_feat.append(math.log(1 - p1) + math.log(prior1))

	return prob_0_per_feat, prob_1_per_feat, priorminus_0_per_feat, priorminus_1_per_feat


# @clock
def logprob(prob_list, pminus_list, lfi, example):
	return sum(prob_list[f] if f in example.feats else pminus_list[f] for f in range(1, lfi+1))


def test_naive_bayes(examples, lfi, false_list, true_list, pminusfalse_list, pminustrue_list):
	num_errors = 0
	print('testing naive bayes')
	for i, example in enumerate(examples):
		print('Example: {}'.format(i))
		# print(example)
		# print('here')
		ptrue = logprob(true_list, pminustrue_list, lfi, example)
		pfalse = logprob(false_list, pminusfalse_list, lfi, example)
		# ptrue = math.log(sum(true_list[f] for f in example.feats) + sum(1-true_list[f] for f in range(1,lfi+1) if f not in example.feats))
		# pfalse = math.log(sum(false_list[f] for f in example.feats) + sum(1 - false_list[f] for f in range(1, lfi+1) if f not in example.feats))
		# num_true = sum(1 for f in example.feats if true_list[f] > false_list[f])
		# num_false = sum(1 for f in example.feats if true_list[f] <= false_list[f])
		# num_false = len(example.feats) - num_true
		# num_false = sum(1 for f in example.feats if f in false_list)
		if ptrue > pfalse and example.label == -1:
			num_errors += 1
		elif pfalse >= ptrue and example.label == 1:
			num_errors += 1
		# if num_true > num_false and example.label == -1:
		# 	num_errors += 1
		# elif num_true < num_false and example.label == 1:
		# 	num_errors += 1
	print('Naive Bayes Acc:\t{}'.format(str(1 - (num_errors / len(examples)))))
	return 1 - (num_errors / len(examples))



def parse_featlist(file_name, num_feats):
	examples = []
	with open(file_name, 'r') as fn:
		for line in fn:
			linsp = line.split()
			label = int(linsp[0])
			feats = [int(feat.split(":")[0]) for feat in linsp[1:]]
			examples.append(LabeledEx(label, feats))

	return examples



def run_naive_bayes(example_sets, num_sets, largest_index):

	# each 'i' is test
	gamma_dict = defaultdict(int)
	for i in range(len(example_sets)):
		# each 'j' is training
		training = []
		for j in range(len(example_sets)):
			if i == j:
				continue
			training.extend(example_sets[j])

		gamma = [2, 1.5, 1, .5]


		print('fold:\t{}'.format(str(i)))
		for gam in gamma:
			print('gamma=\t{}'.format(gam))
			p0, p1, pminus0, pminus1 = naive_bayes(training, gam, lfi)
			acc = test_naive_bayes(example_sets[i], lfi, p0, p1, pminus0, pminus1)
			gamma_dict[gam] += acc

	print('CV averages:')
	for gam in gamma:
		print("Gamma:\t{}\tacc:\t{}".format(gam,gamma_dict[gam]/num_sets))

if __name__ == '__main__':
	test = "data/speeches.test.liblinear"
	training_whole = "data/speeches.train.liblinear"

	base_cvsplits = "data/CVSplits/training0{}.data"

	lfi = max(get_largest_index(training_whole), get_largest_index(test))

	training_examples = [parse_featlist(base_cvsplits.format(i), lfi) for i in range(5)]

	run_naive_bayes(training_examples, len(training_examples), lfi)

	examples = parse_featlist(training_whole, lfi)
	p0, p1, pminus0, pminus1 = naive_bayes(examples, 1, lfi)
	# test_naive_bayes(examples, lfi, p0, p1)

	test_examples = parse_featlist(test, lfi)
	print('naive bayes on whole training')
	test_naive_bayes(examples, lfi, p0, p1, pminus0, pminus1)

	print('naive bayes on test')
	test_naive_bayes(test_examples, lfi, p0, p1, pminus0, pminus1)
import numpy as np
import random
from collections import namedtuple, defaultdict
import math

LabeledEx = namedtuple('LabeledEx', ['label', 'feats'])


def likelihood(dd, feat, label, gamma, count_label):
	return (dd[feat] + gamma) / (count_label + 2*gamma)


def naive_bayes(examples, gamma, lfi):
	dd_0 = defaultdict(int)
	dd_1 = defaultdict(int)
	num_pos = 0
	for example in examples:
		example_feats = [feat for feat in example.feats if feat != 0]
		if example.label == 1:
			for ef in example_feats:
				dd_1[ef] += 1
			num_pos += 1
		else:
			for ef in example_feats:
				dd_0[ef] += 1

	prob_1_per_feat = []
	prob_0_per_feat = []
	for i in range(lfi):

		p0 = likelihood(dd_0, i, 0, gamma, len(examples)-num_pos)
		prior0 = ((len(examples)-num_pos)/len(examples))
		prob_0_per_feat.append(p0 * prior0)

		p1 = likelihood(dd_1, i, 1, gamma, num_pos)
		prior1 = (num_pos / len(examples))
		prob_1_per_feat.append(p1 * prior1)
	return prob_0_per_feat, prob_1_per_feat

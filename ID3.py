"""
ID3 with large number of binary features
"""
from clockdeco import clock
import math
import random
from collections import namedtuple, defaultdict
from learning_util import get_largest_index

LabeledEx = namedtuple('LabeledEx', ['label', 'feats'])

BINARY_LIST = [0,1]
Y_VALS = [1, -1]

"""
features are integers
"""
# @clock
def gain(examples, feature, example_entropy):
	true_in = [lbl for lbl in examples if feature in lbl.feats]
	false_in = [lbl for lbl in examples if feature not in lbl.feats]

	total_gain = (len(true_in) / len(examples)) * entropy(true_in)
	total_gain += (len(false_in) / len(examples)) * entropy(false_in)

	return example_entropy - total_gain


def entropy(examples):
	total = len(examples)
	if total == 0:
		return 0
	sum_pos = sum(1 for lbl in examples if lbl.label == Y_VALS[0])
	sum_neg = sum(1 for lbl in examples if lbl.label == Y_VALS[1])
	p_pos = sum_pos / total
	p_neg = sum_neg / total
	if p_pos == 0:
		if p_neg == 0:
			return 0
		return - p_neg * math.log2(p_neg)
	if p_neg == 0:
		if p_pos == 0:
			return 0
		return -p_pos * math.log2(p_pos)
	return -p_pos * math.log2(p_pos) - p_neg * math.log2(p_neg)


#### ID3 and DECISION TREE ####

"""
Nodes are nested dictionaries with 3 values, a feature, a 0 : {}, and a 1 : {}
"""

def num_samples_with_label(samples, target_label):
	return sum(1 for lbl in samples if lbl.label == target_label)


def all_samples_target(samples, target_label):
	return len(samples) == num_samples_with_label(samples, target_label)

# @clock
def best_feature(examples, feature_list):
	best = (-1, None)
	entropy_examples = entropy(examples)
	for feature in feature_list:
		x = gain(examples, feature, entropy_examples)
		if x > best[0]:
			best = (x, feature)
	return best[1]


def most_labeled(samples, target_labels):
	best = (-1, None)
	for tlabel in target_labels:
		num_with_label = num_samples_with_label(samples, tlabel)
		if num_with_label > best[0]:
			best = (num_with_label, tlabel)
	return best[1]



# ID3 with option to limit depth
@clock
def ID3_depth(examples, features, depth):

	for tlabel in [1, -1]:
		if all_samples_target(examples, tlabel):
			return tlabel

	if len(features) == 0:
		# return most common value of remaining examples
		return most_labeled(examples, Y_VALS)

	# Pick Best Feature (integer
	if len(features) == 1:
		best_f = list(features)[0]
	else:
		best_f = best_feature(examples, features)

	children = {'feature': best_f}

	sub_samples_0 = [exmpl for exmpl in examples if best_f not in exmpl.feats]
	if len(sub_samples_0) == 0 or depth == 1:
		children[0] = most_labeled(sub_samples_0, Y_VALS)
	else:
		children[0] = ID3_depth(sub_samples_0, set(features) - {best_f}, depth-1)

	sub_samples_1 = [exmpl for exmpl in examples if best_f in exmpl.feats]

	if len(sub_samples_1) == 0 or depth == 1:
		children[1] = most_labeled(sub_samples_1, Y_VALS)
	else:
		children[1] = ID3_depth(sub_samples_1, set(features) - {best_f}, depth-1)

	return children


def use_tree(tree, item):

	# base case, the tree is a value
	if type(tree) is bool:
		return tree

	# otherwise, recursively evaluate item with features
	result = tree['feature'] in item.feats
	return use_tree(tree[result], item)


def parse(file_name, num_feats):
	examples = []
	with open(file_name, 'r') as fn:
		for line in fn:
			linsp = line.split()
			label = int(linsp[0])
			feats = [int(feat.split(":")[0]) for feat in linsp[1:]]
			examples.append(LabeledEx(label, feats))

	return examples


def get_x(examples, x):
	z = list(examples)
	random.shuffle(z)
	return z[:x]


def get_x_trees(examples, lfi):
	dtrees = []
	for i in range(1000):
		sub_set = get_x(training_whole, 100)
		dtree = ID3_depth(sub_set, range(lfi), 3)
		dtrees.append(dtree)
	return dtrees

if __name__ == '__main__':
	# read input
	base_cvsplits = "data/CVSplits/training0{}.data"
	training_whole = "data/speeches.train.liblinear"

	test = "data/speeches.test.liblinear"

	# // largest index
	lfi = max(get_largest_index(training_whole), get_largest_index(test))
	examples = parse(training_whole, lfi)

	# training_examples = [parse(base_cvsplits.format(i), lfi) for i in range(5)]
	dtrees = []
	for i in range(1000):
		sub_set = get_x(examples, 100)
		dtree = ID3_depth(sub_set, range(lfi), 3)
		dtrees.append(dtree)

	with open("dtrees.txt", 'w') as dtree_file:
		for dtree in dtrees:
			dtree.write(str(dtree))
			dtree.write('\n')

from collections import namedtuple


LabeledEx = namedtuple('LabeledEx', ['label', 'feats'])


def get_largest_index(training_whole):
	largest = 0
	with open(training_whole, 'r') as tw:
		for line in tw:
			linsp = line.split()
			cndt = int(linsp[-1].split(":")[0])
			if cndt > largest:
				largest = cndt
	return largest



# def parse(file_name, num_feats, ZERO_PAD=0):
# 	examples = []
# 	with open(file_name, 'r') as fn:
# 		for line in fn:
# 			linsp = line.split()
# 			label = int(linsp[0])
# 			feats = [feat.split(":")[0] for feat in linsp[1:]]
# 			feat_vec = np.zeros(num_feats + ZERO_PAD)
# 			for f in feats:
# 				feat_vec[f] = 1.0
# 			examples.append(LabeledEx(label, feat_vec))
#
# 	return examples

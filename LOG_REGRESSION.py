import numpy as np
import random
from collections import namedtuple
import math

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
		p = 1/(1 + exp(example.label * np.transpose(weights) * example.feats))
		tradeoff = (-4 * weights) / (math.sqrt(sigma_squared) ** 3)
		weights = p + learn_rate * tradeoff
		learn_rate = initial_learning_rate / (1 + epoch)
	return weights
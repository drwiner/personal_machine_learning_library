

def get_largest_index(training_whole):
	largest = 0
	with open(training_whole, 'r') as tw:
		for line in tw:
			linsp = line.split()
			cndt = int(linsp[-1].split(":")[0])
			if cndt > largest:
				largest = cndt
	return largest





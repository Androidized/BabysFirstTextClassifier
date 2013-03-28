import random

print("validation set")
with open('data/train.dat', 'r') as trainFile:
	total = 0
	lines = trainFile.readlines()
	random.shuffle(lines)
	halfway = len(lines)/2
	for line in lines[:halfway]:
		word_counts = line.split()
		rating = float(word_counts[-1].split(':')[1])
		total += rating
	avg = int(total / halfway)
	print("\tavg: %i" % avg)
	
	misclassified = 0
	for line in lines[halfway:]:
		word_counts = line.split()
		rating = float(word_counts[-1].split(':')[1])
		if avg != rating:
			misclassified += 1
	print("\tmisclassified: %i" % misclassified)
	print("\tnumber of examples: %i" % halfway)
	acc = float(misclassified) / halfway
	print("\tclassification accuracy: %f" % acc)

with open('data/avg-submission.csv', 'w') as out:
	for n,line in enumerate(file('data/test.dat').readlines()):
		out.write("%i,%i\n" % (n+1,avg))

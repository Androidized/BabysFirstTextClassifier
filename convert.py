#########################################################
# convert the dataset to svmlight format:				#
# - labels should be the very first thing on the line 	#
# - each feature should be an integer id				#
# - features on each line should be in increasing order	#
#########################################################

# words to ids
feature_ids = {}

# returns a list of strings of sorted feature ids to word counts
def convert(docs, train=False):
	docs_formatted = []
	for doc in docs:
		if train:
			label = doc.pop()
		# <feature_id>:<feature_value>
		word_counts_formatted = {}
		for word,count in doc:
			if word not in feature_ids:
				continue
			feat_id = feature_ids[word]
			word_counts_formatted[feat_id] = count
		word_counts_sorted = [str(fid) + ':' + str(c) for fid,c in sorted(word_counts_formatted.items())]
		if train:
			word_counts_sorted.insert(0, label)
		docs_formatted.append(' '.join(word_counts_sorted))
	return docs_formatted

# gathers features from the documents (if not train, includes only features already present)
# returns a list of word-count tuples (and a label at the end if train)
def gather_features(docs, train=False):
	global feature_ids
	old_feature_ids = feature_ids
	feature_ids = {}
	docs_formatted = []
	curr_index = 0

	for doc in docs:
		word_counts = doc.split()
		if train:
			label = str(int(float(word_counts.pop().split(':')[1])))
		
		# <word>:<feature_value>
		word_counts_formatted = []
		for wc in word_counts:
			# there's a couple badly formatted words...just ignore them
			try:
				word, count = wc.split(':')
			except ValueError:
				print('Found a word without a count')
				continue
	
			# in train, include all features
			if train:
				if word not in feature_ids:
					feature_ids[word] = curr_index
					curr_index += 1
			# in test, include only features present in training as well
			else:
				if word not in old_feature_ids:
					continue
				elif word not in feature_ids:
					feature_ids[word] = curr_index
					curr_index += 1
		
			# append a word-count tuple for EZ iterating l8r
			word_counts_formatted.append((word,int(count)))
		if train:
			word_counts_formatted.append(label)
		docs_formatted.append(word_counts_formatted)
	return docs_formatted

with open('data/train.dat') as trainfile, open('data/test.dat') as testfile:
	print('Gathering features...')
	train = gather_features(trainfile.readlines(), train=True)
	test = gather_features(testfile.readlines())
	print('Converting train...')
	train = convert(train, train=True)
	print('Converting test...')
	test = convert(test)
	with open('data/train-svmlight.dat', 'w') as out:
		out.write('\n'.join(train))
	with open('data/test-svmlight.dat', 'w') as out:
		out.write('\n'.join(test))
with open('data/all-features.txt', 'w') as out:
	out.write('\n'.join([word + ',' + str(idx) for word, idx in sorted(feature_ids.items())]))
print('# features: %i' % len(feature_ids))


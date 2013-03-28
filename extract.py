#########################################################
# Extract features from the dataset.					#
# - Binarize the features for presence-based features	#
# - Use TF-IDF for word normalization					#
# - Select the best features using chi2					#
#########################################################

from optparse import OptionParser
import cPickle as pickle

from numpy import *
from scipy.sparse import lil_matrix, csr_matrix, hstack, vstack
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.decomposition import NMF

op = OptionParser()
op.add_option('--binarize', action='store_true', help='Convert features to 0 or 1 for the absence of presence of the feature')
op.add_option('--tfidf', action='store_true', help='Convert values to TF-IDF')
op.add_option('--nmf', action='store', help='Add topics features using Non-negative matrix factorization')
op.add_option('--select-features', action='store', dest='select_features', help='Use only the most relevant features (options: k-best or pct)')
op.add_option('--k', action='store', dest='k_features', help='Number of features or percentile of the highest scores')
(opts, args) = op.parse_args()
if len(args) > 0:
	op.error('This script takes no arguments.')
	sys.exit(1)

# load the data
print('Loading train data...')
X, y = load_svmlight_file('data/train-svmlight.dat')
_, n_features = X.get_shape()

print('Loading test data...')
with open('data/test-svmlight.dat') as infile:
	lines = infile.readlines()
	n_samples = len(lines)
	test = lil_matrix((n_samples, n_features))
	for n,line in enumerate(lines):
		for word_count in line.split():
			fid, count = word_count.split(':')
			test[n,int(fid)] = int(fid)
test = test.tocsr()

if opts.binarize:
	print('Binarizing the data...')
	binar = Binarizer(copy=False)
	X = binar.transform(X)
	test = binar.transform(test)

if opts.tfidf:
	print('Transforming word occurrences into TF-IDF...')
	tranny = TfidfTransformer()
	X = tranny.fit_transform(X)
	test = tranny.transform(test)

if opts.select_features:
	k_features = int(opts.k_features)
	if opts.select_features == 'k-best':
		print('Selecting %i best features...' % k_features)
		ch2 = SelectKBest(chi2, k=k_features)
	if opts.select_features == 'pct':
		print('Selecting features in the top %i percentile...' % k_features)
		ch2 = SelectPercentile(chi2, percentile=k_features)
	X = ch2.fit_transform(X, y)
	test = ch2.transform(test)
	# only allow NMF if using a subset of features (really slow to compute)
	if opts.nmf:
		print('Adding topics features using NMF...')
		n_topics = int(opts.nmf)
		nmf = NMF(n_components=n_topics)
		full_set = vstack([X,test])
		feats = csr_matrix(nmf.fit_transform(full_set))
		X_feats = feats[:X.shape[0],:]
		test_feats = feats[X.shape[0]:,:]
		X = hstack([X,X_feats], format='csr')
		test = hstack([test,test_feats], format='csr')
		print('\a'*5)

print('Saving extracted features...')
triplet = (X, y, test)
with open('data/extracted-features.pkl', 'wb') as out:
	pickle.dump(triplet, out)


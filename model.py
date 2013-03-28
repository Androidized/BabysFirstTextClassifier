#####################################################
# Build individual models.							#
# - Fiddle with parameters using grid search		#
# - Or run CV on all classifiers to compare results	#
#####################################################

from util import clfs, generate_submission

from optparse import OptionParser
import cPickle as pickle

from sklearn.semi_supervised import LabelSpreading
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

op = OptionParser()
op.add_option('--fiddle', action='store_true', help='Fiddle with the parameters of a single model using grid search')
(opts, args) = op.parse_args()
if len(args) > 0:
	op.error('This script takes no arguments.')
	sys.exit(1)

print('Loading data...')
with open('data/extracted-features.pkl', 'rb') as inpkl:
	X, y, test = pickle.load(inpkl)

if opts.fiddle:
	print('Fiddlin wif this classifier...')
	clf = LogisticRegression(fit_intercept=False, C=0.02)
	params = { 
			 }
	# -1 means parallelize it on as many cores as you got
	gs_clf = GridSearchCV(clf, params, n_jobs=-1)
	gs_clf.fit(X, y)
	best_params, score, _  = max(gs_clf.grid_scores_, key=lambda x: x[1])
	print(score)
	print(best_params)
else:
	max_acc = 0
	best_clf = None
	# clfs comes from util
	for clf in clfs:
		scores = cross_val_score(clf, X, y, cv=3)
		print(clf)
		avg_score = scores.mean()
		print("Accuracy: %0.5f (+/- %0.2f)\n" % (avg_score, scores.std() / 2))
		if avg_score > max_acc:
			max_acc = avg_score
			best_clf = clf
	print("Best individual classifier:")
	print(best_clf)
	best_clf.fit(X, y)
	preds = best_clf.predict(test)
	generate_submission(preds)


from util import clfs, generate_submission

import sys
from os.path import exists
from optparse import OptionParser
import cPickle as pickle

from numpy import *
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.utilities import percentError

op = OptionParser()
op.add_option('--ann', action='store_true', help='Use an artificial neural network to blend the models')
op.add_option('--ensemble', action='store_true', help='Use an ensemble classifier to blend the models')
(opts, args) = op.parse_args()
if len(args) > 0:
	op.error('This script takes no arguments.')
	sys.exit(1)
	
print('Gathering cross validation predictions from models to use as dataset...')
with open('data/extracted-features.pkl', 'rb') as inpkl:
	X, y, test = pickle.load(inpkl)
skf = StratifiedKFold(y, n_folds=3)
# iterate over k folds
k = 1
n_clfs = len(clfs)
for train_idx, test_idx in skf:
	print('\tFold %i...' % k)
	
	X_train, y_train = X[train_idx], y[train_idx]
	X_test, y_test = X[test_idx], y[test_idx]
	n_test_samples = len(test_idx)
	
	# save the cross validation predictions of each classifier
	curr_preds = zeros((n_test_samples, n_clfs))
	for n,clf in enumerate(clfs):
		clf.fit(X_train, y_train)
		curr_preds[:,n] = clf.predict(X_test)
	# if it's the first fold, we gotta initiate the matrices
	if k == 1:
		blend_inputs = curr_preds
		blend_targets = y_test
	# otherwise, append the predictions
	else:
		blend_inputs = append(blend_inputs, curr_preds, axis=0)
		blend_targets = append(blend_targets, y_test, axis=0)
	k+=1

print('Gathering full predictions from models...')
full_preds = zeros((test.shape[0],n_clfs))
for n,clf in enumerate(clfs):
	clf.fit(X, y)
	full_preds[:,n] = clf.predict(test)

if opts.ann:
	# prevent PyBrain index error by pretending there are 5 classes
	ds = ClassificationDataSet(n_clfs, 1, nb_classes=5)
	# we gotta reshape, PyBrain expects each label to be on its own row
	# PyBrain requires classes starting from zero (?!?)
	blend_targets = reshape(blend_targets - 1, (blend_targets.shape[0], 1))
	# check that they have the right dimensions before moving on
	assert(blend_inputs.shape[0] == blend_targets.shape[0])
	ds.setField('input', blend_inputs)
	ds.setField('target', blend_targets)
	trainDS, testDS = ds.splitWithProportion(0.25)
	trainDS._convertToOneOfMany()
	testDS._convertToOneOfMany()

	print('Training neural network...')
	net = buildNetwork(trainDS.indim, 5, trainDS.outdim, outclass=SoftmaxLayer) 
	trainer = BackpropTrainer(net, dataset=trainDS, momentum=0.1, weightdecay=0.01)
	# do 20 iterations of 5 epochs each (total 100 epochs)
	for n_iter in range(20):
		trainer.trainEpochs(5)
		trnresult = percentError(trainer.testOnClassData(), trainDS['class'])
		tstresult = percentError(trainer.testOnClassData(dataset=testDS), testDS['class'])
		print("train error: %5.2f%%" % trnresult)
		print("test error: %5.2f%%" % tstresult)
	predDS = ClassificationDataSet(n_clfs, 1, nb_classes=5)
	full_preds = full_preds - 1
	predDS.setField('input', full_preds)
	# uselessly set the targets as all zeros. we won't use these
	predDS.setField('target', zeros((test.shape[0],1)))
	predDS._convertToOneOfMany()
	
	print('Blending...')
	out = net.activateOnDataset(predDS).argmax(axis=1) + 1
	if any(out == 3):
		print('There are some 3s, sir.')

elif opts.ensemble:
	blenders = [
				GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, min_samples_leaf=35, max_features=4),
				RandomForestClassifier(n_estimators=500, max_depth=5, min_density=0.5, oob_score=True),
				ExtraTreesClassifier()
				]
	max_acc = 0
	best_blend = None
	for blender in blenders:
		scores = cross_val_score(blender, blend_inputs, blend_targets, cv=3)
		print(blender)
		avg_score = scores.mean()
		print("Accuracy: %0.5f (+/- %0.2f)\n" % (avg_score, scores.std() / 2))
		if avg_score > max_acc:
			max_acc = avg_score
			best_blend = blender
	print("Best individual classifier:")
	print(best_blend)
	best_blend.fit(blend_inputs, blend_targets)
	out = best_blend.predict(full_preds)

generate_submission(out)

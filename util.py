from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier, LogisticRegression, RandomizedLogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

clfs = [
		MultinomialNB(alpha=0.02), 
		LinearSVC(C=0.5, loss='l2', penalty='l1', dual=False, tol=1e-3),
		RidgeClassifier(tol=1e-2, solver="lsqr"),
		Perceptron(n_iter=50), 
		PassiveAggressiveClassifier(n_iter=50), 
		SGDClassifier(alpha=.0001, n_iter=50, penalty='l1'),
		SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"),
		NearestCentroid(),
		LogisticRegression(fit_intercept=False, C=0.02),
		]

def generate_submission(preds):
	print('\n---GENERATING A SUBMISSION---')
	with open('data/submission.csv', 'w') as out:
		out.write('\n'.join([str(n+1) + ',' + str(int(p)) for n,p in enumerate(preds)]))


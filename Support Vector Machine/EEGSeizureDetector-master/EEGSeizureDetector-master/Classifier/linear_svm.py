import sys

sys.path.insert(0, '/Users/graemecox/Documents/Capstone/Code/eegSvm/PreprocessingFunctions')
sys.path.insert(0, '/Users/graemecox/Documents/Capstone/Code/eegSvm/Features')

from fileReader import *

from sklearn import datasets, svm, metrics, tree, neighbors, neural_network
from sklearn import naive_bayes, linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import cPickle
import numpy as np 

# fn = '/Users/graemecox/Documents/Capstone/Code/eegSvm/Data/labels.txt'
# UPDATE_FEAT = 1
# CREATE_CLASSIFIER = 1
CLASSIFIER_NAME = 'savedClassifier.pkl'

CLASSIFIERS = [
	svm.SVC(kernel='linear', C=0.035),
	tree.DecisionTreeClassifier(max_depth=5),
	neighbors.KNeighborsClassifier(3),
	neural_network.MLPClassifier(alpha=1),
	naive_bayes.GaussianNB(),
	svm.LinearSVC(),
	LinearDiscriminantAnalysis(solver='svd', store_covariance=True),
	QuadraticDiscriminantAnalysis(store_covariances=True),
	# GaussianProcessClassifier(1.0*RBF(1.0)),
	RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	AdaBoostClassifier()
	# tree.DecisionTreeClassifier(max_depth=5, n_estimators=10, max_features=1)
	]
# root = '/Users/graemecox/Documents/Capstone/Code/eegSvm/'




def createLinearClassifier(X_train, y_train,save=0):

	# model = svm.SVC(C=25, kernel='rbf',degree=1)
	model = svm.SVC(C=25, kernel='rbf',degree=1)
	# Data stuff
	y_train.ravel()

	model.fit(X_train,y_train)
	print("Classifier has been created")

	if save:
	#Save classifier
		with open(CLASSIFIER_NAME,'wb') as fid:
			cPickle.dump(model, fid)
		print("Classifier has been saved")

	return model

def crossValidationStuff(model, X_test, y_test, cv_num):
	scores = cross_val_score(model, X_test, y_test, cv=cv_num)
	print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() *2))



def train_multiple_classifiers(X_train, y_train, X_test, y_test, save=0):
	model  = []
	scores = []
	for i in range(len(CLASSIFIERS)):
		
		temp_model = CLASSIFIERS[i]
		print('%%%%%%%%%%%%  Classifier %d: %s' % ((1+i),CLASSIFIERS[i]))

		temp_model.fit(X_train, y_train)
		scores.append(temp_model.score(X_test,y_test))

		print('Score: %f' % scores[i])

	print('------------ FINISHED -----------')
	return scores


def optimizeSVM(X_train, y_train, X_test, y_test):
	parameters= {'kernel':['linear','rbf'],
	'C':[1,2,3,4,5,6,7,8,9,10],
	'gamma':[0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5]}

	# parameters = [
 #  		{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
 #  		{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 # 	]

	# parameters = {'kernel':('linear', 'rbf'),
	#  'C':[1,2,3,4,5,6,7,8,9,10],
	#  'gamma':[0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5]}

	clf = svm.SVC()
	grid = GridSearchCV(clf,parameters)
	grid.fit(X_train, y_train)
	predicted = grid.predict(X_test)
	print('------ Confusion Matrix -----\n%s' % metrics.confusion_matrix(y_test,predicted))

    # print('----- Confusion Matrix ------\n%s'%metrics.confusion_matrix(y_test, predicted ))

def parameterScore(X_train, y_train, X_test, y_test):
	
	# parameters= {'kernel':['linear','rbf'],
	# 'C':[1,2,3,4,5,6,7,8,9,10],
	# 'gamma':[0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5]}

	parameters= {'kernel':['linear','rbf'],
	'C':[1,2,3,4,5,6,7,8,9,10],
	'gamma':[0.01,0.05,0.10,0.20,0.30]}
	scores = ['precision','recall']
	# parameters= {'kernel':['linear','rbf'],
	# 'C':[1,5,10],
	# 'gamma':[0.01,0.5]}
	# scores = ['precision','recall']



	for score in scores:
		print('##### Tuning hyper-parameters for %s\n' % score)

		clf = GridSearchCV(svm.SVC(),  parameters, cv=5, scoring='%s_macro'%score)
		clf.fit(X_train, y_train)

		print("Best parameters set found on development set:\n")
		print(clf.best_params_)
		print("\nGrid scores on devlopment set:\n")
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r" % (mean, std*2, params))

		print("\nClassification Report:\n")
		y_true, y_pred = y_test, clf.predict(X_test)
		print(classification_report(y_true,y_pred))





# labels = np.load('../Data/labels.npy')
# feat = np.load('../Data/features.npy')
# n_classes = 2

# # y = label_binarize(labels, classes=[-1, 1])
# # n_classes = y.shape[1]

# # random_state = np.random.RandomState(0)
# # n_samples, n_features = feat.shape
# # feat = np.c_[feat, random_state.randn(n_samples, 200 * n_features)]

# X_train, X_test, y_train, y_test = train_test_split(
# 	feat, labels, test_size=0.2, random_state=0)

# print('Starting classification')
# # classifier = svm.SVC(C=5, kernel='linear',degree=1)
# # y_score = classifier.fit(X_train, y_train)


# classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
#                                  random_state=random_state))
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# print('Done Classification. Now plotting ROC curve')
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])


# print(X_train.shape)

# print('Start optimize')
# # optimizeSVM(X_train, y_train, X_test, y_test)
# parameterScore(X_train, y_train, X_test, y_test)

# scores = train_multiple_classifiers(X_train, y_train, X_test, y_test,0)
# scores = np.array(scores)
# print('Classifier With the Highest Score: %s \nScore was: %s' % 
# 	(CLASSIFIERS[scores.argmax()], scores[scores.argmax()]))

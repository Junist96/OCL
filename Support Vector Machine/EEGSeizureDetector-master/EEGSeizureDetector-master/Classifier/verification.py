import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt

def crossValidationStuff(model, X_test, y_test, cv_num):
	scores = cross_val_score(model, X_test, y_test, cv=cv_num)
	print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() *2))
	return scores

# def computeROC(X_test,y_test, y_score):
# 	"""
# 		Function to help find the false-positive and false-negative rates
# 	"""
# 	# n_classes= 2 # seizure, no-seizure
# 	n_classes = y_test.shape
# 	print(n_classes)
# 	fpr = dict()
# 	tpr = dict()
# 	roc_auc = dict()
# 	# for i in range(n_classes):
# 		# fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[i])
# 		# roc_auc[i] = auc(fpr[i],tpr[i])

# 	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# 	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 	plt.figure()
# 	lw = 2
# 	plt.plot(fpr[2], tpr[2], color='darkorange',
# 	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# 	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# 	plt.xlim([0.0, 1.0])
# 	plt.ylim([0.0, 1.05])
# 	plt.xlabel('False Positive Rate')
# 	plt.ylabel('True Positive Rate')
# 	plt.title('Receiver operating characteristic example')
# 	plt.legend(loc="lower right")
# 	plt.show()

# def plotROC(fpr, tpr):
# 	print(fpr)




def computeStats(model, X_test, y_train, y_test,cv = 5):

	score = crossValidationStuff(model, X_test, y_test, cv=cv)
	# score = crossValidationStuff(model, X_test, y_test, cv=5)
	# probs = model.predict_proba(X_test)
	# # preds = probs[:,1]
	# print(probs.shape)


def classifierStats(clf, X_test, y_test):
	pred = clf.predict(X_test) # get list of predictions for comparison

	print('----- Printing Classification Report ----- \n%s'%metrics.classification_report(y_test, pred, target_names=['Ictal','Interictal']))
	print('----- Confusion Matrix ------\n%s'%metrics.confusion_matrix(y_test, pred))


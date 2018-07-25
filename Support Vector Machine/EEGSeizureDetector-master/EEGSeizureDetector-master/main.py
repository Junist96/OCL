import sys
sys.path.insert(0, 'PreprocessingFunctions')
sys.path.insert(0, 'Classifier')

from kaggleReader import *
from linear_svm import *

from sklearn import datasets, svm, metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
import numpy as np

root = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/'

# feat,labels, test_clips = readKaggleDataset(root)


CREATE_CLASSIFIER = 1
UPDATE_FEAT = 1

def main():


## Feature Extraction Stage
# Either load the feature or re-find them
	if (UPDATE_FEAT):
		## Read from CSV
		# labels, feat = readCsv('/Datafeatures.csv',hasHeader=0)
		# print('Read in labels and features from csv')

		feat, labels, test_clips = readKaggleDataset(root,saveFiles=1)
	else:
		labels = np.load('Data/labels.npy')
		feat = np.load('Data/features.npy')
		print('Loaded labels and features')

	#Read in Kaggle labels and features
	X_train, X_test, y_train, y_test = train_test_split(
	feat, labels, test_size=0.2, random_state=0)


## Classification
# Either load data 
	if (CREATE_CLASSIFIER):
		print('Creating classifier now. Sit tight!')
		model = createLinearClassifier(X_train, y_train, save=1)
	else:
		with open(CLASSIFIER_NAME,'rb') as fid:
			clf = cPickle.load(fid)
		print('Loaded classifier')

	crossValidationStuff(model, X_test, y_test,5)

if __name__ == "__main__":
	main()
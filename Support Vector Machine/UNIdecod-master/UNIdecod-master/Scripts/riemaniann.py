from __future__ import division
import sys as sys
import itertools
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from mne.decoding import Vectorizer
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,  accuracy_score
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import (ERPCovariances, XdawnCovariances,
                                  HankelCovariances)
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sp
from os import path
import os

#--------------------- FUNCTION TO PLOT CONFUSION MATRICES ------------------------------

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "%.1f" % round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#--------------------- FUNCTION TO DECODE USING A SUPPORT VECTOR MACHINE ------------------------------


def SVM_decoding_on_full_epochs(X,y,plot_conf_matrix=0,class_names=None,test_size=0.2,n_splits=5):
    """ This function decodes on the full epoch using standard SVM algorithm

    Parameters
    ---------
    X : data extracted from the epochs provided to the decoder
    y : categorical variable (i.e. discrete but it can be more then 2 categories)
    plot_confusion_matrix : set to 1 if you wanna see the confusion matrix
    class_names: needed for the legend if confusion matrices are plotted ['cat1','cat2','cat3']
    test_size : proportion of the data on which you want to test the decoder
    n_splits : when calculating the score, number of cross-validation folds

    Returns:
    -------
    score, y_test, y_pred

    """

    # ------- define the classifier -------
    scaler = preprocessing.StandardScaler()
    vectorizer = Vectorizer()
    clf = SVC(C=1,kernel='linear',decision_function_shape ='ovr')
    concat_classifier = Pipeline([('vector',vectorizer),('scaler',scaler),('svm',clf)])

    # This returns the 5 scores calculated for each fold

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y = np.asarray(y)
    scores = []
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train on X_train, y_train
        concat_classifier.fit(X_train, y_train)
        # Test on X_test and then score
        y_pred = concat_classifier.predict(X_test)
        scores.append(accuracy_score(y_true=y_test,y_pred=y_pred))
    scores = np.asarray(scores)


    if plot_conf_matrix==1:
        print('you chose to plot the confusion matrix')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7, stratify=y)
        y_pred = concat_classifier.fit(X_train, y_train).predict(X_test)

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=3)
        print(cnf_matrix)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')

        plt.show()
        return scores, y_test, y_pred, cnf_matrix

    return scores, cnf_matrix

def my_pipeline(scores, y_test, y_pred, cnf_matrix, code, fb_fold):

	# indicate opened fold
    if fb_fold:
        sys.stdout.write("<")
        sys.stdout.flush()

    if code == 'Xdawn_':
    	results = {
	        'scores_Xdawn': scores,
	        'y_test_Xdawn': y_test,
	        'y_pred_Xdawn': y_pred,
	        'cnf_matrix_Xdawn': cnf_matrix}
        
    elif code == 'Hankel':
	        results = {
	        'scores_Hankel': scores,
	        'y_test_Hankel': y_test,
	        'y_pred_Hankel': y_pred,
	        'cnf_matrix_Hankel': cnf_matrix}
	    
    elif code == 'ERPcov':
	            results = {
	        'scores_ERPcov': scores,
	        'y_test_ERPcov': y_test,
	        'y_pred_ERPcov': y_pred,
	        'cnf_matrix_ERPcov': cnf_matrix}
    
    print('results dictionnary created')
    # indicate end of fold
    if fb_fold:
        sys.stdout.write(">")
        sys.stdout.flush()
    return results

#--------------------- FUNCTION TO DECODE USING A RIEMANIANN SUPPORT VECTOR MACHINE ------------------------------


def pyR_decoding_on_full_epochs(X,y,plot_conf_matrix=0,class_names=None,test_size=0.2,n_splits=5,classifier='ERP_cov'):
    """ This function decodes on the full epoch using the pyRiemannian decoder
    cf https://github.com/Team-BK/Biomag2016/blob/master/Final_Submission.ipynb

    Parameters
    ---------
    X : data extracted from the epochs provided to the decoder
    y : categorical variable (i.e. discrete but it can be more then 2 categories)
    plot_confusion_matrix : set to 1 if you wanna see the confusion matrix
    class_names: needed for the legend if confusion matrices are plotted ['cat1','cat2','cat3']
    test_size : proportion of the data on which you wanna test the decoder
    n_splits : when calculating the score, number of cross-validation folds
    classifier : set it to 'ERP_cov', 'Xdawn_cov' or 'Hankel_cov' depending on the classification you want to do.

    Returns: scores, y_test, y_pred, cnf_matrix or just scores if you don't want the confusion matrix
    -------

    """

    # ------- define the classifier -------
    if classifier=='ERP_cov':
        spatial_filter = UnsupervisedSpatialFilter(PCA(20), average=False)
        ERP_cov = ERPCovariances(estimator='lwf')
        CSP_30 = CSP(30, log=False)
        tang = TangentSpace('logeuclid')
        clf = make_pipeline(
            spatial_filter,ERP_cov,
            CSP_30,
            tang,
            LogisticRegression('l2'))

    if classifier == 'Xdawn_cov':
        clf = make_pipeline(
        UnsupervisedSpatialFilter(PCA(50), average=False),
        XdawnCovariances(12, estimator='lwf', xdawn_estimator='lwf'),
        TangentSpace('logeuclid'),
        LogisticRegression('l2'))

    if classifier == 'Hankel_cov':
        clf = make_pipeline(
            UnsupervisedSpatialFilter(PCA(70), average=False),
            HankelCovariances(delays=[1, 8, 12, 64], estimator='oas'),
            CSP(15, log=False),
            TangentSpace('logeuclid'),
            LogisticRegression('l2'))


    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=4343)
    y = np.asarray(y)
    scores = []
    for train_index, test_index in cv.split(X,y):
        print(train_index)
        print(test_index)
        print('we are in the CV loop')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train on X_train, y_train
        clf.fit(X_train, y_train)
        # Predict the category on X_test
        y_pred = clf.predict(X_test)

        scores.append(accuracy_score(y_true=y_test,y_pred=y_pred))
    scores = np.asarray(scores)


    if plot_conf_matrix==1:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7, stratify=y)
        print('train and test have been split')
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        print(cnf_matrix)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')

        plt.show()
        return scores, y_test, y_pred, cnf_matrix

    return scores, y_test, y_pred, cnf_matrix


# ======================================================================================================================================
# ====================================== LET'S TRY THE PREVIOUS FUNCTIONS  =============================================================
# ======================================================================================================================================

# Localizer on spatial positions. The subject sees the 8 positions on the vertices of the octagone. We want to decode the presented location.

# we load the epochs and the corresponding labels
# MEG data recorded with Elekta Neuromag system, 306 channels, sampling frequency 1000Hz decimated 20 times.
# Each dot is shown during 100 ms, SOA 433 ms between the dots. The time window on which the epochs are built goes from -100 ms to 450 ms, where 0ms is dot presentation.
results = {}
 
file=sp.loadmat('riemaniann.mat')

#epochs = np.load(data_name)
epochs = file['Xm']
# look at the shape of the data : it is n_epochs X n_channels X n_times (192 trials per position)
epochs.shape

# load the labels corresponding to the positions
#labels = np.load(labels_name)
labels = file['y']
labels = np.squeeze(labels)

code=file['code']
results_path=str(file['results_path'][0])

# let's now try the different decoders.
class_names = ['pos0','pos1']
fb_fold = True

if code == 'Xdawn_':
    scores_Xdawn, y_test_Xdawn, y_pred_Xdawn, cnf_matrix_Xdawn = pyR_decoding_on_full_epochs(epochs,labels,plot_conf_matrix=1,class_names=class_names,test_size=0.2,n_splits=5,classifier='Xdawn_cov')
    results = my_pipeline(scores_Xdawn, y_test_Xdawn, y_pred_Xdawn, cnf_matrix_Xdawn, code=code, fb_fold=fb_fold)
    file_name= results_path + 'scores_XDawn.mat'
    
    
elif code == 'Hankel':
    scores_Hankel, y_test_Hankel, y_pred_Hankel, cnf_matrix_Hankel = pyR_decoding_on_full_epochs(epochs,labels,plot_conf_matrix=1,class_names=class_names,test_size=0.2,n_splits=5,classifier='Hankel_cov')
    results = my_pipeline(scores_Hankel, y_test_Hankel, y_pred_Hankel, cnf_matrix_Hankel, code=code, fb_fold=fb_fold)
    file_name= results_path + 'scores_Henkel.mat'

elif code == 'ERPcov':
    scores_ERPcov, y_test_ERPcov, y_pred_ERPcov, cnf_matrix_ERPcov = pyR_decoding_on_full_epochs(epochs,labels,plot_conf_matrix=1,class_names=class_names,test_size=0.2,n_splits=5,classifier='ERP_cov')
    results = my_pipeline(scores_ERPcov, y_test_ERPcov, y_pred_ERPcov, cnf_matrix_ERPcov, code=code, fb_fold=fb_fold)
    file_name= results_path + 'scores_ERPcov.mat'


else : 
	print ('No analysis performed. Please check the "code" variable is either "Xdawn_","Hankel" or "ERPcov", then run the script again.')

sp.savemat(file_name, results)

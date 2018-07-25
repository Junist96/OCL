##kaggle reader
import sys
sys.path.insert(0, '/Users/graemecox/Documents/Capstone/Code/eegSvm/Features')

from fe_wavelet import *
from fe_bandpower import *

import numpy as np 
import scipy.io as spio
import os
import glob



# fn = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1/Dog_1_ictal_segment_1.mat'
num_feat = 5

def printMatData(fn):
	mat = spio.loadmat(fn)

	print(mat['freq'])
	print(mat['latency'])
	print(np.array(mat['data']).shape)

	## Matlab structure fields:
	sampling_freq = mat['freq']
	latency = mat['latency']

	data = np.array(mat['data'])

	print(data.shape)

def getSubfolders(root):
	subfolders = os.listdir(root)
	subfolders.remove('.DS_Store')
	return subfolders
	# for folder in subfolders:
	# 	print(root+folder)
	# 	fns = os.listdir(root+folder)
	# 	print(len(fns))



def returnFeatures(fn):

	mat = spio.loadmat(fn)

	Fs = mat['freq']
	data = np.array(mat['data'])

	num_elec = data.shape[0]

	features = np.empty((0,num_feat))

	for i in range(num_elec):
		elec_data = data[i][:]

		# meanamp = fe_freqbandmean(elec_data,Fs)
		# w_db4 = fe_waveletdecomp(elec_data)
		temp_features = fe_waveletdecomp(elec_data)
		# temp_features = fe_freqbandmean(elec_data,Fs)
		# temp_features = fe_spectralratio(elec_data,Fs)


		# append all features vertically
		# temp_features = np.concatenate((meanamp,w_db4), axis=1)
		
		#Append new features to next row in feature list
		features = np.append(features,temp_features,axis=0)

	return features

def returnLabels(feat, label):
	temp_size = feat.shape[0]
	labels = np.empty((temp_size,), int)
	labels[0:temp_size] = label #Ictal is -1
	return labels



def readKaggleDataset(root,saveFiles=0):
	subfolders = getSubfolders(root)

	interictal_clips = []
	ictal_clips = []
	test=[];
	test_clips = []

	labels = []
	# num_feat = 10

	features = np.empty((0,num_feat), float)

	print('----------------STARTING TO READ FROM SUBFOLDERS----------------')
	for subfolder in subfolders:
		# print(root+subfolder)
		folder = root+subfolder
		print('Reading data from subfolder  %s' % folder)

		files = os.listdir(folder) #Get all files in subfolder

		#Find all files with certain keyword
		for file in glob.glob(folder+'/*_ictal_*.mat'):
			ictal_clips.append(file)

			#Get features
			temp_feat = returnFeatures(file)
			temp_labels = returnLabels(temp_feat, -1)
			#Append labels to labels list

			# Append features and labels
			labels = np.append(labels, temp_labels, axis=0)
			features = np.append(features, temp_feat, axis=0)



		for file in glob.glob(folder+'/*_interictal_*.mat'):
			interictal_clips.append(file)

			#Get features

			temp_feat = returnFeatures(file)
			temp_labels = returnLabels(temp_feat,1)

			# Append features and labels
			labels = np.append(labels, temp_labels, axis=0)
			features = np.append(features, temp_feat, axis=0)

		for file in glob.glob(folder+'/*_test_*.mat'):
			test_clips.append(file)

	print('Length of interictal clips is %d'% len(interictal_clips))
	print('Length of ictal clips is %d'% len(ictal_clips))
	print('Length of test clips is %d'% len(test_clips))

	print('----------------Finished extracting features----------------')

	if saveFiles:
		np.save('Data/labels.npy', labels)
		print('Saved labels as Data/labels.npy')
		np.save('Data/features.npy', features)
		print('Saved features as Data/features.npy')


	return features, labels, test_clips


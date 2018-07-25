import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import scipy
from pylab import *

freq_bands = {'Delta': (0,4),
			'Theta': (4,8),
			'Alpha': (8,12),
			'Beta': (12,30),
			'Gamma': (30,45)
			}

def fe_meanAmp(fn):


	mat = spio.loadmat(fn)

	Fs = mat['freq']
	data = np.array(mat['data'])

	num_elec = data.shape[0]

	features = []
	for i in range(num_elec):
		elec_data = data[i][:]



		amps=np.absolute(np.fft.rfft(elec_data))
		freqs = np.fft.rfftfreq(len(elec_data),1.0/Fs)

		# Take the mean of the fft amplitude for each EEG band
		temp_band = dict()
		for band in freq_bands:  
			#Find all indexs that belong to each EEG frequency band
		    i = np.where((freqs >= freq_bands[band][0]) & 
		                       (freqs <= freq_bands[band][1]))[0]
		    temp_band[band] = np.mean(amps[i])

		# print(L)
		# features = [][]
		values = [temp_band['Delta'],
			temp_band['Theta'],
			temp_band['Alpha'],
			temp_band['Beta'],
			temp_band['Gamma']]

		features.append(values)
		# print(np.array(values).shape)


	features = np.array(features)
	return features

	# for i in range(num_elec):
		# elec_data = data[i][:]

def fe_freqbandmean(data, Fs):
		amps=np.absolute(np.fft.rfft(data))
		freqs = np.fft.rfftfreq(len(data),1.0/Fs)

		# Take the mean of the fft amplitude for each EEG band
		temp_band = dict()
		for band in freq_bands:  
			#Find all indexs that belong to each EEG frequency band
		    i = np.where((freqs >= freq_bands[band][0]) & 
		                       (freqs <= freq_bands[band][1]))[0]
		    temp_band[band] = np.mean(amps[i])

		# print(L)
		# features = [][]
		values = [temp_band['Delta'],
			temp_band['Theta'],
			temp_band['Alpha'],
			temp_band['Beta'],
			temp_band['Gamma']]



		return np.reshape(np.array(values),(-1,5)) #Make sure we return a 2d array, not 1d

def fe_spectralratio(data,Fs):
		"""
		Returns ratio of mean frequencies of all data


		"""

		amps=np.absolute(np.fft.rfft(data))
		freqs = np.fft.rfftfreq(len(data),1.0/Fs)

		# Take the mean of the fft amplitude for each EEG band
		temp_band = dict()
		for band in freq_bands:  
			#Find all indexs that belong to each EEG frequency band
		    i = np.where((freqs >= freq_bands[band][0]) & 
		                       (freqs <= freq_bands[band][1]))[0]
		    temp_band[band] = np.mean(amps[i])

		# print(L)
		# features = [][]
		band_mean = [temp_band['Delta'],
			temp_band['Theta'],
			temp_band['Alpha'],
			temp_band['Beta'],
			temp_band['Gamma']]


		values= np.empty(len(band_mean)**2,)
		count = 0


		band_mean= np.array(band_mean)
		band_mean[np.isnan(band_mean)] = 0

		for i in range(len(band_mean)):
			
			for j in range(len(band_mean)):	
				# if band_mean[j] == 0:
					# values[count] = 
				values[count] = band_mean[i]/band_mean[j]
				count = count+1
		# Get rid of all NaN values in array


		return np.reshape(values,(-1,25)) #Make sure we return a 2d array, not 1d


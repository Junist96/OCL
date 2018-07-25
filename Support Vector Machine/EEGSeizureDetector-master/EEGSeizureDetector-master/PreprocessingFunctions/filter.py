## functions related to noise and filtering

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import scipy
from pylab import *



def addNoise(data):
	maxNoise = 10 # mV
	len_noise = len(data)

	noise = maxNoise * np.random.normal(0,1,len_noise)
	#0 is the mean of the normal distribution
	#1 is the Standard deviation of normal distribution

	return data+noise



def plotComparePureAndNoise(data,Fs):
	t = np.arange(len(data))*(1/Fs)

	mag = np.absolute(np.fft.rfft(data))
	freqs = np.fft.rfftfreq(len(data), 1.0/Fs)



	f, axarr = plt.subplots(2,2)

	axarr[0,0].plot(t, data)
	axarr[0,0].set_title('Time Series of Original Data')

	axarr[0,1].plot(freqs, mag)
	axarr[0,1].set_title('Fourier Series of Original Data')

	noise_data = addNoise(data)
	noise_mag = np.absolute(np.fft.rfft(noise_data))

	axarr[1,0].plot(t,noise_data)
	axarr[1,0].set_title('Time Series of Noisy Data')

	axarr[1,1].plot(freqs, noise_mag)
	axarr[1,1].set_title('Fourier Series of Noisy Data')

	plt.show()


# fn = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1/Dog_1_ictal_segment_1.mat'



# mat = spio.loadmat(fn)
# # print(mat)

# print(mat['freq'])
# print(mat['latency'])
# print(np.array(mat['data']).shape)

# ## Matlab structure fields:
# sampling_freq = mat['freq']
# latency = mat['latency']

# data = np.array(mat['data'])

# elec_data = data[1][:]
# print(elec_data.shape)

# plotComparePureAndNoise(elec_data, sampling_freq)

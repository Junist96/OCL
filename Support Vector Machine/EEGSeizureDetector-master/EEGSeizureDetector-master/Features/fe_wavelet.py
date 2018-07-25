## Wavelet feature extraction

import numpy as np
import pywt
import scipy.io as spio
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def mean(data):
	return sum(data)/float(len(data))

def quickFFT(data):
	Y = np.fft.fft(data)
	L = len(data)
	P2 = np.abs(Y/L)

	f_seiz = P2[0:L/2+1]
	f_seiz[1:L/2+1] = 2*f_seiz[1:L/2+1]
	return f_seiz


def fe_wavelet(fn):

	mat = spio.loadmat(fn)


	## FFT  
	sampling_freq = mat['freq']
	# latency = mat['latency']

	data = np.array(mat['data'])

	#Get number of electrodes
	num_elec = data.shape[0]

	wname = 'db4';

	w = pywt.Wavelet(wname)
	features = []
	for i in range(num_elec):
		elec_data = data[i][:]

		filt = np.convolve(elec_data, w.dec_lo)
		filt_D4 = np.convolve(filt, w.dec_hi)

		f_seiz = quickFFT(filt_D4)
		# print(L)
		# features = [][]
		values = [mean(f_seiz[49:75]),
			mean(f_seiz[75:100]),
			mean(f_seiz[124:150]),
			mean(f_seiz[149:175]),
			mean(f_seiz[174:200])]

		features.append(values)
		# print(np.array(values).shape)


	features = np.array(features)
	return features

def fe_waveletdecomp(data, wname='db4'):
		w = pywt.Wavelet(wname)
		filt = np.convolve(data, w.dec_lo)
		filt_D4 = np.convolve(filt, w.dec_hi)

		f_seiz = quickFFT(filt_D4)

		values = [mean(f_seiz[49:75]),
			mean(f_seiz[75:100]),
			mean(f_seiz[124:150]),
			mean(f_seiz[149:175]),
			mean(f_seiz[174:200])]

		return np.reshape(np.array(values),(-1,5))

## Test stuff
# test = np.empty((0,5), float)
# test2 = fe_wavelet(fn)

# test = np.append(test,test2, axis=0)
# test = np.append(test,test2, axis=0)
# print(test.shape)
# print(np.concatenate(test,test2))

# print(test.shape)


# ## FFT  
# sampling_freq = mat['freq']
# latency = mat['latency']



# num_elec = data.shape[0]

# wname = 'db4';

# w = pywt.Wavelet(wname)	
# elec_data = data[0][:]


# ##Should be final lenght 417. (assuming db4 and initial 400)

# cA, cD = pywt.dwt(elec_data, wname)

# print(len(cA))
# print(cA[0])
# print(cA[1])


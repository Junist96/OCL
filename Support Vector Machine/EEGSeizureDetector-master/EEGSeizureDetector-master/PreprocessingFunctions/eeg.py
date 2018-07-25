import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import os



class EEG_Sample:

	def __init__(self, path):
		self.fn = path

		mat = spio.loadmat(path)
		self.Fs = mat['freq']
		self.latency = mat['latency'] #doesn't show up in test data
		self.data = mat['data']
		self.numElec = self.data.shape[0]
		tempFolder = np.array(fn.split('/'))
		self.subject = tempFolder[len(tempFolder) - 2] #Specifies which subject the data is from

		# self.label = 
		if (-1 != path.find('_ictal_')):
			self.label = -1
		elif (-1 != path.find('_interictal_')):
			self.label = 1
		else:
		 	self.label = 0 # for test data. 



	def plotTimeSeries(self, elecNum):
		if (elecNum > self.numElec):
			print('Electrode number not valid for this dataset')
			return

		elec_data = self.data[elecNum-1][:]
		t = np.arange(len(elec_data)) * (1/self.Fs)

		plt.plot(t,elec_data)
		plt.xlabel('Time (s)')
		plt.ylabel('Voltage (mV)')
		plt.title('Time Series of Electrode #%d' % elecNum)
		plt.show()


	def plotFourierSeries(self, elecNum):
		if (elecNum > self.numElec):
			print('Electrode number not valid for this dataset')
			return
		elec_data = self.data[elecNum-1]
		mag = np.absolute(np.fft.rfft(elec_data))

		freqs = np.fft.rfftfreq(len(elec_data), 1.0/self.Fs)

		plt.plot(freqs,mag)
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('Magnitude')
		plt.title('Fourier Series of Electrode #%d' % elecNum)
		plt.show()


	# def addNoise(data):


# fn = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1/Dog_1_ictal_segment_1.mat'

# test = EEG_Sample(fn)
# print(test.subject)
# test.plotFourierSeries(60)
# test.plotTimeSeries(17)

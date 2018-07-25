## Script for reading in either text file or csv
import numpy as np
import csv


# fn= 'testData.csv'



def readCsv(fn, hasHeader=0):
	##Reads in .csv specified in input, and outputs labels and features
	# Labels are in the first column, and features are in the other columns
	#is header defines whetehr the columns have names or not

	f = open(fn,'rb')
	# f = open(sys.argv[1], 'rb')

	if (hasHeader):

		print("Don't know what the header looks like")

	else:
		data = []
		reader = csv.reader(f)
		for row in reader:
			data.append(row)
		data= np.array(data)

	f.close()

	#Labels are in the first column
	labels = data[:,0]

	#Features are all other columns
	features= data[:,1:]
	return labels, features


def readTextFile(fn,saveFile=0, saveName=None):
	# Specifies to read from text file
	# fn specifies which file is supplied
	#saveFile toggles whether to save as a numpy array to quicikly load at a later time
	data = []
	with open(fn) as f:
		for line in f:
			#Split each line by comma, then save as a list of floats
			currentline = [float(x) for x in line.split(",")]
			data.append(currentline)
	f.close()
	data = np.array(data)

	if data.shape[1] == 1:
		tempName = 'labels_py'
	else:
		tempName = 'features_py'
	saveName = tempName if saveName is None else saveName


	if saveFile == 1:
		np.save(saveName,data)

	return data

# lab, feat = readCsv(fn,0)

# print(lab.shape)
# print(feat.shape)


# fn = '/Users/graemecox/Documents/Capstone/Code/eegSvm/labels.txt'
# data = readTextFile(fn,1,'test')

# print(data.shape)


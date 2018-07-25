## feature selection script
import numpy as np 
from sklearn.decomposition import PCA

def use_pca(features, numComp=3):
	pca = PCA(n_components=numComp)
	fit = pca.fit(features)

	print('PCA Explained Variance: %s' % fit.explained_variance_ratio_)
	return fit.components_


x = [[1,0,3,2] ,[1,2,3,4]]
use_pca(x,3)
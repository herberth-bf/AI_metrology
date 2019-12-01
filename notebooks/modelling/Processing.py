# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:37:30 2019

Processing METHODS


@author: Herberth Frohlich
"""
import os
import statsmodels.api as sm
from sklearn.decomposition import PCA

class Processing:
    
	def __init__(self):
		pass
		
	def lowess_smooth(self, vector, frac=0.05):
		lowess = sm.nonparametric.lowess
		FRAC = 0.05 # fraction for lowess
		pass
		
		
		
class FeatureReduction:
	def __init__(self):
		pass
				
	def pca_(self, vectorTrain, vectorTest, vectorVal, n):
		pca = PCA(svd_solver = 'full', n_components=n)
		pca.fit(vectorTrain)
		X_train_pca = pca.transform(vectorTrain)
		X_val_pca = pca.transform(vectorVal)
		X_test_pca = pca.transform(vectorTest)
		return X_train_pca, X_test_pca, X_val_pca, pca
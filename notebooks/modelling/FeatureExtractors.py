# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:12:38 2019

# FEATURE EXTRACTORS 

@author: Herberth Frohlich
"""
import numpy as np
import pandas as pd
from skimage.feature import hog

class FeatureExtractors:
    
	def __init__(self):
		pass
    
	def deviations(self, image):
		matrix = np.zeros((2, image.shape[1]))
		matrix[0,:] = np.std(image, axis=1) 
		matrix[1,:] = np.std(image, axis=0) 
		matrix = matrix.reshape(1, -1)
		return matrix
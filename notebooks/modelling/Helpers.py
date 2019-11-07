# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:12:38 2019

# FEATURE EXTRACTORS 

@author: Herberth Frohlich
"""
import glob
import re
import numpy as np
import pandas as pd
import cv2

class Loaders:
    
	def __init__(self):
		pass
    
	def load_images(self, inputPath, dim):
		images = []
		files = sorted(glob.glob(os.path.join(inputPath, "*.bmp")))
		files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[0]))
		for f in files:
			im = cv2.imread(f, 0)
			resized = cv2.resize(im, dim)
			images.append(resized)
		return np.array(images), files
		
		
class Metrics:

	def __init__(self):
		pass

	def mean_absolute_percentage_error(self, y_true, y_pred): 
		y_true, y_pred = np.array(y_true), np.array(y_pred)
		return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
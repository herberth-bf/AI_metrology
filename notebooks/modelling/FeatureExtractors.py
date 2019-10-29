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
    
    def stdev_method(self, image):
        
        pass
    
    
    def hog_extraction(self, image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=False):
        
        fd, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=visualize, multichannel=multichannel)
		
        return fd, hog_image
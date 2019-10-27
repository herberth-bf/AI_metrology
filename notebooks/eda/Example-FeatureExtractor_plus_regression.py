# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:36:14 2019

EXAMPLE FEATURE EXTRACTOR PLUS REGRESSION FOR SHEAROGRAPHY IMAGES

OBJECTIVES: check if regression is possible using image as inputs.
check if the interpolation and extracpolation are possible with a trained model.


@author: Herberth Frohlich
"""
import os
import glob
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
sys.path.insert(1, r'D:\Users\Herberth Frohlich\Documents\AI_shearo\notebooks\modelling')

# Loading Shearography Images

trainAttrX = pd.read_csv(r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\TrainSet\trainEnergies.txt", header=None)
testAttrX = pd.read_csv(r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\TestSet\testEnergies.txt", header=None)
valAttrX= pd.read_csv(r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\ValSet\valEnergies.txt", header=None)
dim = (329, 329)
def load_images(inputPath,dim):
    images = []
    files = glob.glob(os.path.join(inputPath, "*.bmp"))
    for f in files:
        im = cv2.imread(f, 0)
        resized = cv2.resize(im, dim)
        images.append(resized)
    return np.array(images)


images_train = load_images(r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\TrainSet", dim)
trainImagesX = images_train / 255.0
#trainImagesX = trainImagesX.reshape(-1, dim[0], dim[1], 1)

images_test = load_images(r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\TestSet", dim)
testImagesX = images_test / 255.0
#testImagesX = testImagesX.reshape(-1, dim[0], dim[1], 1)

images_val = load_images(r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\ValSet", dim)
valImagesX  = images_val / 255.0
#valImagesX = valImagesX.reshape(-1, dim[0], dim[1], 1)


image = trainImagesX[500, :, :]
cv2.imshow("", image)


vertical = np.std(image, axis=1) 
horizontal = np.std(image, axis=0) 
# TODO -  use this two vectors as features for regression straight forward.
plt.plot(vertical)

# Applying standard deviatio method
#from FeatureExtractors import FeatureExtractors
#feat = FeatureExtractors()




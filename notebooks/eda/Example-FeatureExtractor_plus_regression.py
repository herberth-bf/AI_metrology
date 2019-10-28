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
valAttrX = pd.read_csv(r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\ValSet\valEnergies.txt", header=None)
dim = (329, 329)
def load_images(inputPath,dim):
    images = []
    files = glob.glob(os.path.join(inputPath, "*.bmp"))
    labels = os.listdir(inputPath)
    for f in files:
        im = cv2.imread(f, 0)
        resized = cv2.resize(im, dim)
        images.append(resized)
    return np.array(images), labels

# TODO CONSERTAR ORDEM DE CARREGAMENTO DOS ARQUIVOS DAS PASTAS

images_train, files_train = load_images(r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\TrainSet", dim)
trainImagesX = images_train# / 255.0
#trainImagesX = trainImagesX.reshape(-1, dim[0], dim[1], 1)

images_test, files_test = load_images(r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\TestSet", dim)
testImagesX = images_test# / 255.0
#testImagesX = testImagesX.reshape(-1, dim[0], dim[1], 1)

images_val, files_val = load_images(r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\ValSet", dim)
valImagesX  = images_val# / 255.0
#valImagesX = valImagesX.reshape(-1, dim[0], dim[1], 1)

image = valImagesX[0, :, :]
cv2.imshow("", image)

def deviations(image):
    matrix = np.zeros((2, image.shape[1]))
    matrix[0,:] = np.std(image, axis=1) 
    matrix[1,:] = np.std(image, axis=0) 
    matrix = matrix.reshape(1, -1)
    return matrix

#matrix_example = deviations(image)
#plt.plot(np.transpose(matrix_example))


# Acquisition of features via deviation of rows and columns
X_train = np.empty((0, dim[0]*2))
for i in range(trainImagesX.shape[0]):
    im = trainImagesX[i, :, :]
    X_train = np.append(X_train, deviations(im), axis=0)
 
X_val = np.empty((0, dim[0]*2))
for i in range(valImagesX.shape[0]):
    im = valImagesX[i, :, :]
    X_val = np.append(X_val, deviations(im), axis=0)
 
X_test = np.empty((0, dim[0]*2))
for i in range(testImagesX.shape[0]):
    im = testImagesX[i, :, :]
    X_test = np.append(X_test, deviations(im), axis=0)

maxEnergy = trainAttrX[0].max()     
y_train = trainAttrX[0] / maxEnergy
y_test = testAttrX[0] / maxEnergy
y_val = valAttrX[0] / maxEnergy

## calling regression
#from RegressionMethods import RegressionMethods
#reg = RegressionMethods()
#    
#%%   
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
reg = RandomForestRegressor(n_estimators=5, oob_score=True, 
                            verbose=0, max_depth=3,
                            min_samples_split=3)    
reg.fit(X_train, y_train)    
y_pred = reg.predict(X_val)
valMAE = mean_absolute_error(y_val, y_pred)
valScore = r2_score(y_val, y_pred) 
print(valMAE, valScore)
plt.plot(y_val)
plt.plot(y_pred)

# Applying standard deviatio method
#from FeatureExtractors import FeatureExtractors
#feat = FeatureExtractors()




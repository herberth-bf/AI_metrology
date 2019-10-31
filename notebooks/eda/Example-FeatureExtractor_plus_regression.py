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
import re
import sys
import pandas as pd
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
sys.path.insert(1, r'D:\Users\Herberth Frohlich\Documents\AI_shearo\notebooks\modelling')

# Loading Shearography Images
trainPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\TrainSet")
testPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\TestSet")
valPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\ValSet")


trainAttrX = pd.read_csv(os.path.join(trainPath, "trainEnergies.txt"), header=None)
testAttrX = pd.read_csv(os.path.join(testPath, "testEnergies.txt"), header=None)
valAttrX = pd.read_csv(os.path.join(valPath, "valEnergies.txt"), header=None)
dim = (128, 128)
cells_per_block = (2, 2)
pixels_per_cell = (24, 24) 
orientations = 8
def load_images(inputPath,dim):
    images = []
    files = sorted(glob.glob(os.path.join(inputPath, "*.bmp")))
    files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[0]))
    for f in files:
        im = cv2.imread(f, 0)
        resized = cv2.resize(im, dim)
        images.append(resized)
    return np.array(images), files

images_train, files_train = load_images(trainPath, dim)
trainImagesX = images_train# / 255.0
#trainImagesX = trainImagesX.reshape(-1, dim[0], dim[1], 1)

images_test, files_test = load_images(testPath, dim)
testImagesX = images_test# / 255.0
#testImagesX = testImagesX.reshape(-1, dim[0], dim[1], 1)

images_val, files_val = load_images(valPath, dim)
valImagesX  = images_val# / 255.0
#valImagesX = valImagesX.reshape(-1, dim[0], dim[1], 1)


# Getting image size template for further matrices initialization
image = valImagesX[157, :, :]
fd_pattern, _ = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, 
                cells_per_block=cells_per_block, visualize=True, multichannel=False,  block_norm ='L1')
#cv2.imshow("", image)
hog_size = len(fd_pattern)
print(hog_size)

def deviations(image):
    matrix = np.zeros((2, image.shape[1]))
    matrix[0,:] = np.std(image, axis=1) 
    matrix[1,:] = np.std(image, axis=0) 
    matrix = matrix.reshape(1, -1)
    return matrix

#matrix_example = deviations(image)
#plt.plot(np.transpose(matrix_example))

# Acquisition of features via deviation of rows and columns and hog at the same time
X_train_hog = np.empty((0, hog_size))
X_train = np.empty((0, dim[0]*2))
for i in range(trainImagesX.shape[0]):
    im = trainImagesX[i, :, :]
    X_train = np.append(X_train, deviations(im), axis=0)
    fd, _ = hog(im, orientations=orientations, pixels_per_cell=pixels_per_cell, 
                cells_per_block=cells_per_block, visualize=True, multichannel=False,  block_norm ='L1')
    X_train_hog = np.append(X_train_hog, fd.reshape(1,hog_size), axis=0)
 
X_val_hog = np.empty((0, hog_size))
X_val = np.empty((0, dim[0]*2))
for i in range(valImagesX.shape[0]):
    im = valImagesX[i, :, :]
    X_val = np.append(X_val, deviations(im), axis=0)
    fd, _ = hog(im, orientations=orientations, pixels_per_cell=pixels_per_cell, 
                cells_per_block=cells_per_block, visualize=True, multichannel=False,  block_norm ='L1')
    X_val_hog = np.append(X_val_hog,  fd.reshape(1, hog_size), axis=0)
 
X_test_hog = np.empty((0, hog_size))
X_test = np.empty((0, dim[0]*2))
for i in range(testImagesX.shape[0]):
    im = testImagesX[i, :, :]
    X_test = np.append(X_test, deviations(im), axis=0)
    fd, _ = hog(im, orientations=orientations, pixels_per_cell=pixels_per_cell, 
                cells_per_block=cells_per_block, visualize=True, multichannel=False,  block_norm ='L1')
    X_test_hog = np.append(X_test_hog, fd.reshape(1,hog_size), axis=0)
 

maxEnergy = trainAttrX[0].max()     
y_train = trainAttrX[0] / maxEnergy
y_test = testAttrX[0] / maxEnergy
y_val = valAttrX[0] / maxEnergy

#%%
# pca
from sklearn.decomposition import PCA

pca = PCA(svd_solver = 'full', n_components=150)
pca.fit(X_train_hog)
X_train_hog_pca = pca.transform(X_train_hog)
X_val_hog_pca = pca.transform(X_val_hog)
X_test_hog_pca = pca.transform(X_test_hog)

cumulative = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumulative)



## calling regression
#from RegressionMethods import RegressionMethods
#reg = RegressionMethods()
#%% 

X_train = X_train
X_test = X_test
X_val= X_val

  
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
reg = RandomForestRegressor(n_estimators=60, oob_score=True, 
                            verbose=0, max_depth=3,
                            min_samples_split=5)    
reg.fit(X_train, y_train)    

# Validation
y_pred = reg.predict(X_val)
valMAE = mean_absolute_error(y_val, y_pred)
valScore = r2_score(y_val, y_pred) 
print(valMAE, valScore)
plt.scatter(np.arange(len(y_val.values)), y_val.values, s=15)
plt.scatter(np.arange(len(y_pred)), y_pred, s=15)

# TO DO IMPLEMENT GRID SEARCH



#%%
# Testing interpolation
y_predT = reg.predict(X_test)
testMAE = mean_absolute_error(y_test, y_predT)
testScore = r2_score(y_test, y_predT) 
print(testMAE, testScore)
plt.scatter(np.arange(len(y_test.values)), y_test.values, s=15)
plt.scatter(np.arange(len(y_predT)), y_predT, s=15)

# Applying standard deviatio method
#from FeatureExtractors import FeatureExtractors
#feat = FeatureExtractors()




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
import statsmodels.api as sm
import matplotlib.pyplot as plt
from skimage.feature import hog

sys.path.insert(1, r'D:\Users\Herberth Frohlich\Documents\AI_shearo\notebooks\modelling')

# Loading Shearography Images
trainPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\TrainSet")
testPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\TestSet")
valPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\ValSet")

trainAttrX = pd.read_csv(os.path.join(trainPath, "trainEnergies.txt"), header=None)
testAttrX = pd.read_csv(os.path.join(testPath, "testEnergies.txt"), header=None)
valAttrX = pd.read_csv(os.path.join(valPath, "valEnergies.txt"), header=None)
dim = (64, 64)
cells_per_block = (2, 2)
pixels_per_cell = (24, 24) 
orientations = 8
lowess = sm.nonparametric.lowess
FRAC = 0.05 # fraction for lowess

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

# making sure they are in crescent order and other preprocessing
maxEnergy = trainAttrX[0].max() 
  
y_train = trainAttrX[0]# / maxEnergy
y_test = testAttrX[0]# / maxEnergy
y_val = valAttrX[0]# / maxEnergy
indxTest = np.argsort(y_test)
indxVal = np.argsort(y_val)

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
#X_train_hog = np.empty((0, hog_size))
X_train = np.empty((0, dim[0]*2))
#X_train_smooth = np.empty((0, dim[0]*2))
for i in range(trainImagesX.shape[0]):
    print("train image {}".format(i))
    im = trainImagesX[i, :, :]
    X_train = np.append(X_train, deviations(im), axis=0)
#    dev_smooth = (lowess(deviations(im).reshape(-1,), np.arange(int(dim[0]*2)), frac=FRAC)).transpose()
#    X_train_smooth = np.append(X_train_smooth, dev_smooth[1,:].reshape(1,-1), axis=0)
#    fd, _ = hog(im, orientations=orientations, pixels_per_cell=pixels_per_cell, 
#                cells_per_block=cells_per_block, visualize=True, multichannel=False,  block_norm ='L1')
#    X_train_hog = np.append(X_train_hog, fd.reshape(1,hog_size), axis=0)
#np.save("X_train_smooth_329_005", X_train_smooth)   
#np.save("X_train_hog_329_2_24_8", X_train_hog)   
    
#X_val_hog = np.empty((0, hog_size))
X_val = np.empty((0, dim[0]*2))
#X_val_smooth = np.empty((0, dim[0]*2))
for i in range(valImagesX.shape[0]):
    print("val image {}".format(i))
    im = valImagesX[i, :, :]
    X_val = np.append(X_val, deviations(im), axis=0)
#    dev_smooth = (lowess(deviations(im).reshape(-1,), np.arange(int(dim[0]*2)), frac=FRAC)).transpose()
#    X_val_smooth = np.append(X_val_smooth, dev_smooth[1,:].reshape(1,-1), axis=0)
#    fd, _ = hog(im, orientations=orientations, pixels_per_cell=pixels_per_cell, 
#                cells_per_block=cells_per_block, visualize=True, multichannel=False,  block_norm ='L1')
#    X_val_hog = np.append(X_val_hog,  fd.reshape(1, hog_size), axis=0)
#np.save("X_val_smooth_329_005", X_val_smooth)
#np.save("X_val_hog_329_2_24_8", X_val_hog)     

#X_test_hog = np.empty((0, hog_size))
X_test = np.empty((0, dim[0]*2))
#X_test_smooth = np.empty((0, dim[0]*2))
for i in range(testImagesX.shape[0]):
    print("test image {}".format(i))
    im = testImagesX[i, :, :]
    X_test = np.append(X_test, deviations(im), axis=0)
#    dev_smooth = (lowess(deviations(im).reshape(-1,), np.arange(int(dim[0]*2)), frac=FRAC)).transpose()
#    X_test_smooth = np.append(X_test_smooth, dev_smooth[1,:].reshape(1,-1), axis=0)
#    fd, _ = hog(im, orientations=orientations, pixels_per_cell=pixels_per_cell, 
#                cells_per_block=cells_per_block, visualize=True, multichannel=False,  block_norm ='L1')
#    X_test_hog = np.append(X_test_hog, fd.reshape(1,hog_size), axis=0)
#np.save("X_test_smooth_329_005", X_test_smooth)
#np.save("X_test_hog_329_2_24_8", X_test_hog)
  

X_train_hog = np.load("X_train_hog_329_2_24_8.npy")  
X_train_smooth = np.load("X_train_smooth_329_005.npy")
X_val_hog = np.load("X_val_hog_329_2_24_8.npy")  
X_val_smooth = np.load("X_val_smooth_329_005.npy")
X_test_smooth = np.load("X_test_smooth_329_005.npy")
X_test_hog = np.load("X_test_hog_329_2_24_8.npy")

#%%
# pca
from sklearn.decomposition import PCA
def pca_(vectorTrain, vectorTest, vectorVal, n=1000):
    pca = PCA(svd_solver = 'full', n_components=n)
    pca.fit(vectorTrain)
    X_train_pca = pca.transform(vectorTrain)
    X_val_pca = pca.transform(vectorVal)
    X_test_pca = pca.transform(vectorTest)
    return X_train_pca, X_test_pca, X_val_pca, pca
#%%
X_train_pca, X_test_pca, X_val_pca, pca = pca_(X_train, X_test, X_val, n=15)
cumulative = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumulative)
## calling regression
#from RegressionMethods import RegressionMethods
#reg = RegressionMethods()
#%% 
tipo = "dev + pca"
X_train_USE = X_train_pca
X_test_USE = X_test_pca
X_val_USE = X_val_pca

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

N_ESTIMATORS = 70 
MAX_DEPTH = 9
MIN_SAMPLES_SPLIT = 2

reg = RandomForestRegressor(n_estimators=N_ESTIMATORS, oob_score=True, 
                            verbose=0, max_depth=MAX_DEPTH,
                            min_samples_split=MIN_SAMPLES_SPLIT, random_state=0)    
reg.fit(X_train_USE, y_train)    

# Validation
y_pred = reg.predict(X_val_USE)
valMAE = mean_absolute_percentage_error(y_val, y_pred)
valScore = r2_score(y_val, y_pred) 
print(valMAE, valScore)
y_val_ = y_val[indxVal]
y_pred = y_pred[indxVal]

# Testing interpolation
y_predT = reg.predict(X_test_USE)
testMAE = mean_absolute_percentage_error(y_test, y_predT)
testScore = r2_score(y_test, y_predT) 
print(testMAE, testScore)
y_test_ = y_test[indxTest]
y_predT = y_predT[indxTest]

fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False)
ax = axs[0]
ax.scatter(np.arange(len(y_val_.values)), y_val_.values, s=5)
ax.scatter(np.arange(len(y_pred)), y_pred, s=15)
ax.set_title("ValSet - MAPE: {}, R2Score: {}".format(np.round(valMAE,3), np.round(valScore,3)))
ax.set_ylabel("Energy[J]")
ax.set_xlabel("Observations")

ax = axs[1]
ax.scatter(np.arange(len(y_test_.values)), y_test_.values, s=5)
ax.scatter(np.arange(len(y_predT)), y_predT, s=15)
ax.set_title("TestSet - MAPE: {}, R2Score: {}".format(np.round(testMAE,3), np.round(testScore,3)))
ax.set_ylabel("Energy[J]")
ax.set_xlabel("Observations")
fig.suptitle("TYPE: {}, IMG DIM: {}, N_ESTIMATORS: {}, MAX_DEPTH: {}, LEAF_SIZE: {} ".format(tipo, dim[0], N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT))
plt.show()

#%%
# TODO GRID SEARCH
# TODO ERROR BAR PLOT
    
    






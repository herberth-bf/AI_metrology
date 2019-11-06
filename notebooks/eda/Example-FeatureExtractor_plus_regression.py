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
from sklearn.linear_model import LinearRegression
from skimage.feature import hog
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(1, r'D:\Users\Herberth Frohlich\Documents\AI_shearo\notebooks\modelling')

# Loading Shearography Images
trainPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\TrainSet")
testPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\TestSet")
valPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\ValSet")

trainAttrX = pd.read_csv(os.path.join(trainPath, "trainEnergies.txt"), header=None)
testAttrX = pd.read_csv(os.path.join(testPath, "testEnergies.txt"), header=None)
valAttrX = pd.read_csv(os.path.join(valPath, "valEnergies.txt"), header=None)

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

# making sure they are in crescent order and other preprocessing
maxEnergy = trainAttrX[0].max() 
y_train = trainAttrX[0]# / maxEnergy
y_test = testAttrX[0]# / maxEnergy
y_val = valAttrX[0]# / maxEnergy
indxTest = np.argsort(y_test)
indxVal = np.argsort(y_val)

# Getting image size template for further matrices initialization
#image = valImagesX[157, :, :]
#fd_pattern, _ = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, 
#               cells_per_block=cells_per_block, visualize=True, multichannel=False,  block_norm ='L1')
#cv2.imshow("", image)
#hog_size = len(fd_pattern)
#print(hog_size)

def deviations(image):
    matrix = np.zeros((2, image.shape[1]))
    matrix[0,:] = np.std(image, axis=1) 
    matrix[1,:] = np.std(image, axis=0) 
    matrix = matrix.reshape(1, -1)
    return matrix

#matrix_example = deviations(image)
#plt.plot(np.transpose(matrix_example))

# Acquisition of features via deviation of rows and columns and hog at the same time

def feature_acquisition(dim):
    
    images_train, files_train = load_images(trainPath, dim)
    trainImagesX = images_train# / 255.0
    #trainImagesX = trainImagesX.reshape(-1, dim[0], dim[1], 1)
    images_test, files_test = load_images(testPath, dim)
    testImagesX = images_test# / 255.0
    #testImagesX = testImagesX.reshape(-1, dim[0], dim[1], 1)
    images_val, files_val = load_images(valPath, dim)
    valImagesX  = images_val# / 255.0
    #valImagesX = valImagesX.reshape(-1, dim[0], dim[1], 1)
    
    #X_train_hog = np.empty((0, hog_size))
    X_train = np.empty((0, dim[0]*2))
    #X_train_smooth = np.empty((0, dim[0]*2))
    for i in range(trainImagesX.shape[0]):
        #print("train image {}".format(i))
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
        #print("val image {}".format(i))
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
        #print("test image {}".format(i))
        im = testImagesX[i, :, :]
        X_test = np.append(X_test, deviations(im), axis=0)
    #    dev_smooth = (lowess(deviations(im).reshape(-1,), np.arange(int(dim[0]*2)), frac=FRAC)).transpose()
    #    X_test_smooth = np.append(X_test_smooth, dev_smooth[1,:].reshape(1,-1), axis=0)
    #    fd, _ = hog(im, orientations=orientations, pixels_per_cell=pixels_per_cell, 
    #                cells_per_block=cells_per_block, visualize=True, multichannel=False,  block_norm ='L1')
    #    X_test_hog = np.append(X_test_hog, fd.reshape(1,hog_size), axis=0)
    #np.save("X_test_smooth_329_005", X_test_smooth)
    #np.save("X_test_hog_329_2_24_8", X_test_hog)
    return X_train, X_val, X_test

#X_train_hog = np.load("X_train_hog_329_2_24_8.npy")  
#X_train_smooth = np.load("X_train_smooth_329_005.npy")
#X_val_hog = np.load("X_val_hog_329_2_24_8.npy")  
#X_val_smooth = np.load("X_val_smooth_329_005.npy")
#X_test_smooth = np.load("X_test_smooth_329_005.npy")
#X_test_hog = np.load("X_test_hog_329_2_24_8.npy")

#%% Helper functions
# pca

from sklearn.decomposition import PCA
def pca_(vectorTrain, vectorTest, vectorVal, n=1000):
    pca = PCA(svd_solver = 'full', n_components=n)
    pca.fit(vectorTrain)
    X_train_pca = pca.transform(vectorTrain)
    X_val_pca = pca.transform(vectorVal)
    X_test_pca = pca.transform(vectorTest)
    return X_train_pca, X_test_pca, X_val_pca, pca

# MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#%%
#X_train_pca, X_test_pca, X_val_pca, pca = pca_(X_train, X_test, X_val, n=15)
#cumulative = np.cumsum(pca.explained_variance_ratio_)
#plt.plot(cumulative)

## calling regression
#from RegressionMethods import RegressionMethods
#reg = RegressionMethods()
#%% 

# GRID SEARCH DEVIATION + PCA
DIM = [(64, 64), (128, 128), (256, 256), (329, 329)] #dimensions for deviations
COMPONENTS = [0.8, 0.9, 1] # in terms of percentage of variance retained
N_ESTIMATORS = [30, 60, 120] # tree parameter
MAX_DEPTH = [3, 6, 12] # tree parameter
MIN_SAMPLES_SPLIT = [3, 4, 5] # tree parameter

numberOfScores = len(DIM)*len(COMPONENTS)*len(N_ESTIMATORS)*len(MAX_DEPTH)*len(MIN_SAMPLES_SPLIT)
#predMatrixFull = [[] for i in range(numberOfScores)]
scoresRF = np.zeros((numberOfScores, 8))
scoresGB = np.zeros((numberOfScores, 8))
scoresVoting = np.zeros((numberOfScores, 8))
count=0

for d in DIM:
    X_train, X_val, X_test = feature_acquisition(d)
    
    for c in COMPONENTS:
        # calculate the maximum number of components for the grid search then keep some of them
        X_train_USE, X_test_USE, X_val_USE, pca = pca_(X_train, X_test, X_val, n=X_train.shape[1])
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        index = np.argmax(cumulative>=c)
        X_train_USE = X_train_USE[:, index:]
        X_val_USE = X_val_USE[:, index:]
        X_test_USE = X_test_USE[:, index:]
        
        for n in N_ESTIMATORS:
            for m in MAX_DEPTH:
                for s in MIN_SAMPLES_SPLIT:
                    
                    #print("Count:{} --> Dimension {}, Component {}, Estimators {}, Max Depth {}, Min Sample Split {}".format(count, d, c, n, m, s))
                    # Fitting the model: random forest
                    reg1 = RandomForestRegressor(n_estimators=n, oob_score=True, 
                            verbose=0, max_depth=m, min_samples_split=s, random_state=0,
                            criterion='mae', max_features='sqrt')    
                    reg1.fit(X_train_USE, y_train)
                    
                    # Fitting the model: gradient boosting
                    reg2 = GradientBoostingRegressor(n_estimators=n, learning_rate=0.5,
                            verbose=0, max_depth=m, min_samples_split=s, random_state=0,
                            criterion='mae', max_features='sqrt')    
                    reg2.fit(X_train_USE, y_train)
                    
                    # Averaged by voting
                    ereg = VotingRegressor([('rf', reg1), ('gb', reg2)])
                    ereg.fit(X_train_USE, y_train)
                    
                    # Validation
                    y_pred_reg1 = reg1.predict(X_val_USE)
                    y_pred_reg2 = reg2.predict(X_val_USE)
                    y_pred_ereg = ereg.predict(X_val_USE)
                    
                    # Scores Random Forest
                    scoresRF[count, 0] = mean_absolute_error(y_val, y_pred_reg1) # MAE
                    scoresRF[count, 1] = mean_absolute_percentage_error(y_val, y_pred_reg1) # MAPE
                    scoresRF[count, 2] = r2_score(y_val, y_pred_reg1) # SCORE
                    scoresRF[count, 3] = d[0]
                    scoresRF[count, 4] = c
                    scoresRF[count, 5] = n
                    scoresRF[count, 6] = m
                    scoresRF[count, 7] = s
                    
                    # Scores Gradient Boosting
                    scoresGB[count, 0] = mean_absolute_error(y_val, y_pred_reg2) # MAE
                    scoresGB[count, 1] = mean_absolute_percentage_error(y_val, y_pred_reg2) # MAPE
                    scoresGB[count, 2] = r2_score(y_val, y_pred_reg2) # SCORE
                    scoresGB[count, 3] = d[0]
                    scoresGB[count, 4] = c
                    scoresGB[count, 5] = n
                    scoresGB[count, 6] = m
                    scoresGB[count, 7] = s
                    
                    # Scores Voting Regression
                    scoresVoting[count, 0] = mean_absolute_error(y_val, y_pred_ereg) # MAE
                    scoresVoting[count, 1] = mean_absolute_percentage_error(y_val, y_pred_ereg) # MAPE
                    scoresVoting[count, 2] = r2_score(y_val, y_pred_ereg) # SCORE
                    scoresVoting[count, 3] = d[0]
                    scoresVoting[count, 4] = c
                    scoresVoting[count, 5] = n
                    scoresVoting[count, 6] = m
                    scoresVoting[count, 7] = s
                               
                    print("Count {} --> MAE RF {}, \ GB {} \ Voting {},".format(count, np.round(scoresRF[count, 0],2), 
                          np.round(scoresGB[count, 0],2), np.round(scoresVoting[count, 0],2)))
                    count+=1

# Save corresponding matrices
np.save("scoresRF_dev+pca+rf", scoresRF)
np.save("scoresGB_dev+pca+rf", scoresGB)
np.save("scoresVoting_dev+pca+rf", scoresVoting)
#np.save("predMatrixFull_dev+pca+rf", predMatrixFull)                   
#%% Retrieving the best model

indexBestModel = np.argmin(scoresVoting[:, 0])
dimBest = (int(scoresVoting[indexBestModel, 3]), int(scoresVoting[indexBestModel, 3]))
cBest = int(scoresVoting[indexBestModel, 4])
nBest = int(scoresVoting[indexBestModel, 5])
mBest = int(scoresVoting[indexBestModel, 6])
sBest = int(scoresVoting[indexBestModel, 7])

X_train, X_val, X_test = feature_acquisition(dimBest)
X_train_USE, X_test_USE, X_val_USE, pca = pca_(X_train, X_test, X_val, n=X_train.shape[1])
cumulative = np.cumsum(pca.explained_variance_ratio_)
index = np.argmax(cumulative>=cBest)
X_train_USE = X_train_USE[:, index:]
X_val_USE = X_val_USE[:, index:]
X_test_USE = X_test_USE[:, index:]
reg1_best = RandomForestRegressor(n_estimators=nBest, oob_score=True, 
                            verbose=0, max_depth=mBest, min_samples_split=sBest, random_state=0,
                            criterion='mae', max_features='sqrt')    
reg2_best = GradientBoostingRegressor(n_estimators=nBest, learning_rate=0.5,
                            verbose=0, max_depth=mBest, min_samples_split=sBest, random_state=0,
                            criterion='mae', max_features='sqrt')    

ereg_best = VotingRegressor([('rf', reg1_best), ('gb', reg2_best)])
ereg_best.fit(X_train_USE, y_train)

import pickle
filename = 'dev+pca+voting.sav'
pickle.dump(ereg_best, open(filename, 'wb'))

#%% Interpolation and testing - model proofing    
#ereg_best = pickle.load(open(filename, 'rb'))           
# Testing interpolation
y_predT = ereg_best.predict(X_test_USE)
testMAE = mean_absolute_error(y_test, y_predT)
testMAPE = mean_absolute_percentage_error(y_test, y_predT)
testScore = r2_score(y_test, y_predT) 
print(testMAE, testMAPE)
y_test_ = y_test[indxTest]
y_predT = y_predT[indxTest]


# TODO SEARCH FOR THE BEST PARAMETERS INSIDE SCORE MATRI

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False)
#ax = axs[0]
#ax.scatter(np.arange(len(y_val_.values)), y_val_.values, s=5)
#ax.scatter(np.arange(len(y_pred)), y_pred, s=15)
#ax.set_title("ValSet - MAPE: {}, R2Score: {}".format(np.round(valMAPE,3), np.round(valScore,3)))
#ax.set_ylabel("Energy[J]")
#ax.set_xlabel("Observations")

tipo = "DEV + PCA +  RF - voting"
ax = axs
ax.scatter(np.arange(len(y_test_.values)), y_test_.values, s=5)
ax.scatter(np.arange(len(y_predT)), y_predT, s=15)
ax.set_title("TestSet - MAE: {}, MAPE: {}".format(np.round(testMAE,3), np.round(testMAPE,3)))
ax.set_ylabel("Energy[J]")
ax.set_xlabel("Observations")
fig.suptitle("TYPE: {}, IMG DIM: {}, N_ESTIMATORS: {}, MAX_DEPTH: {}, LEAF_SIZE: {} ".format(tipo, dimBest[0], nBest, mBest, sBest))
plt.show()

#%%
# TODO GRID SEARCH
# TODO ERROR BAR PLOT
    
    






# USAGE
# python cnn_regression.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages

import os
import cv2
import sys
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
sys.path.insert(1, r'D:\Users\Herberth Frohlich\Documents\AI_shearo\notebooks\modelling')

BATCH_SIZE = 8
LEARNING_RATE = 1e-03

# Loading Shearography Images
trainPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\TrainSet")
testPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\TestSet")
valPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\ValSet")

trainAttrX = pd.read_csv(os.path.join(trainPath, "trainEnergies.txt"), header=None)
testAttrX = pd.read_csv(os.path.join(testPath, "testEnergies.txt"), header=None)
valAttrX = pd.read_csv(os.path.join(valPath, "valEnergies.txt"), header=None)
dim = (32, 32)

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
trainImagesX = trainImagesX.reshape(-1, dim[0], dim[1], 1)

images_test, files_test = load_images(testPath, dim)
testImagesX = images_test# / 255.0
testImagesX = testImagesX.reshape(-1, dim[0], dim[1], 1)

images_val, files_val = load_images(valPath, dim)
valImagesX  = images_val# / 255.0
valImagesX = valImagesX.reshape(-1, dim[0], dim[1], 1)

maxEnergy = trainAttrX[0].max()     
y_train = trainAttrX[0] / maxEnergy
y_test = testAttrX[0] / maxEnergy
y_val = valAttrX[0] / maxEnergy
#%%
#model = models.create_cnn(dim[0], dim[1], 1, filters = (16,32,64), regress=True)
input_shape = (dim[0], dim[1], 1)
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
#model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu')
#model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation="linear"))

opt = Adam(lr=LEARNING_RATE, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
es = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
model.fit(trainImagesX, y_train, 
          validation_data=(valImagesX, y_val),
          epochs=100, 
          callbacks=[es], 
          steps_per_epoch = len(y_train) // BATCH_SIZE, 
          validation_steps = len(y_val) // BATCH_SIZE)

#model.save('cnn_regression.h5')

#%%
#model = load_model('my_model.h5')
# Testing interpolation
from sklearn.metrics import mean_absolute_error, r2_score
y_predT = model.predict(testImagesX)
testMAE = mean_absolute_error(y_test, y_predT)
testScore = r2_score(y_test, y_predT) 
print(testMAE, testScore)
plt.scatter(np.arange(len(y_test.values)), y_test.values, s=15)
plt.scatter(np.arange(len(y_predT)), y_predT, s=15)
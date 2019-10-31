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
from keras.optimizers import Adam, SGD, Adadelta
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
sys.path.insert(1, r'D:\Users\Herberth Frohlich\Documents\AI_shearo\notebooks\modelling')

IMAGE_SIZE = (329,329)
BATCH_SIZE = 2
LEARNING_RATE = 1e-03
NUM_EPOCHS = 100
# Loading Shearography Images
#trainPath = (r"D:\MachineLearningInOptics\NoDecomposition\TrainSet")
#testPath = (r"D:\MachineLearningInOptics\NoDecomposition\TestSet")
#valPath = (r"D:\MachineLearningInOptics\NoDecomposition\ValSet")

trainPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\TrainSet")
testPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\TestSet")
valPath = (r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegressionFull\ValSet")

trainAttrX = pd.read_csv(os.path.join(trainPath, "trainEnergies.txt"), header=None, names = ['score'])
testAttrX = pd.read_csv(os.path.join(testPath, "testEnergies.txt"), header=None, names = ['score'])
valAttrX = pd.read_csv(os.path.join(valPath, "valEnergies.txt"), header=None, names = ['score'])

#create a vector of string *.bmp
def create_string(size):
    strs = ["" for x in range(size)]
    for s in range(size):
        strs[s] = str(s)+".bmp"
    return pd.DataFrame({'id': strs})
        
trainAttrX = trainAttrX.join(create_string(trainAttrX.shape[0]))
testAttrX = testAttrX.join(create_string(testAttrX.shape[0]))
valAttrX = valAttrX.join(create_string(valAttrX.shape[0]))

#train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = True,
#                                   fill_mode = "nearest",
#                                   width_shift_range = 0.2, height_shift_range=0.2,
#                                   rotation_range=30) 

train_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_dataframe(dataframe=trainAttrX, 
                                              directory=trainPath, 
                                              x_col="id", 
                                              y_col="score", 
                                              class_mode="other", 
                                              target_size=(IMAGE_SIZE), 
                                              batch_size=BATCH_SIZE,  
                                              color_mode = "grayscale")


valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_dataframe(dataframe=valAttrX, 
                                              directory=valPath, 
                                              x_col="id", 
                                              y_col="score",
                                              class_mode="other", 
                                              target_size=(IMAGE_SIZE), 
                                              batch_size=BATCH_SIZE,  
                                              shuffle=False, 
                                              color_mode = "grayscale")


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(dataframe=testAttrX, 
                                              directory=testPath, 
                                              x_col="id", 
                                              y_col="score",
                                              class_mode="other", 
                                              target_size=(IMAGE_SIZE), 
                                              batch_size=1,  
                                              shuffle=False, 
                                              color_mode = "grayscale")

#%%
#model = models.create_cnn(dim[0], dim[1], 1, filters = (16,32,64), regress=True)
input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
#model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))
#model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation="linear"))

#opt = Adam(lr=LEARNING_RATE, decay=1e-3 / 200)
#opt = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.95, nesterov=True)
opt = Adadelta()
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
es = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
history = model.fit_generator(train_generator,
                        steps_per_epoch = train_generator.samples // BATCH_SIZE,
                        validation_data = valid_generator,
                        validation_steps = valid_generator.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
                        callbacks=[es])


#model.save('cnn_regression.h5')

#%%
#model = load_model('my_model.h5')
# Testing interpolation
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
probabilities = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
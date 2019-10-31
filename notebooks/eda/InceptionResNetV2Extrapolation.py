"""
This script goes along my blog post:
Keras InceptionResetV2 (https://jkjung-avt.github.io/keras-inceptionresnetv2/)
"""

import pandas as pd
import numpy as np
import time
import os
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout, Activation
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.python.keras.optimizers import Adam, Adadelta,SGD
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import load_model

DATASET_PATH  = r'D:\MachineLearningInOpticsExtrapolation\NoDecomposition'
IMAGE_SIZE    = (329,329)
BATCH_SIZE    = 2  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 550  # freeze the first this many layers for training
NUM_EPOCHS    = 100
WEIGHTS_FINAL = 'model-inception_resnet_v2-NoDecompositionRegression.h5'
DENSE = 64
PATIENCE = 2
LEARNING_RATE = 1e-03

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
                                              color_mode = "rgb")


valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_dataframe(dataframe=valAttrX, 
                                              directory=valPath, 
                                              x_col="id", 
                                              y_col="score",
                                              class_mode="other", 
                                              target_size=(IMAGE_SIZE), 
                                              batch_size=BATCH_SIZE,  
                                              shuffle=False, 
                                              color_mode = "rgb")


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(dataframe=testAttrX, 
                                              directory=testPath, 
                                              x_col="id", 
                                              y_col="score",
                                              class_mode="other", 
                                              target_size=(IMAGE_SIZE), 
                                              batch_size=1,  
                                              shuffle=False, 
                                              color_mode = "rgb")

#%%
# build our classifier model based on pre-trained InceptionResNetV2:
# 1. we don't include the top (fully connected) layers of InceptionResNetV2
# 2. we add a DropOut layer followed by a Dense (fully connected)
#    layer which generates softmax class score for each class
# 3. we compile the final model using an Adam optimizer, with a
#    low learning rate (since we are 'fine-tuning')
net = InceptionResNetV2(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(DENSE)(x)
x = Activation("relu")(x)
#####################
#output_layer = Dense(NUM_CLASSES, activation='sigmoid', name='sigmoid')(x)
output_layer = Dense(1, activation='linear', name='sigmoid')(x)
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True
#net_final.compile(optimizer=Adam(lr=1e-5),
#                  loss='categorical_crossentropy', metrics=['accuracy'])
#net_final.compile(optimizer=Adam(lr=1e-6),
#                  loss='mean_squared_error', metrics=['mse'])
#opt = Adadelta()
opt = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.95, nesterov=True)
net_final.compile(optimizer=opt, loss='mean_absolute_percentage_error')  

#%%
es = EarlyStopping(monitor='val_loss', verbose=1, patience=PATIENCE)
# train the model
start_time = time.time()
history = net_final.fit_generator(train_generator,
                        steps_per_epoch = train_generator.samples // BATCH_SIZE,
                        validation_data = valid_generator,
                        validation_steps = valid_generator.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
                        callbacks=[es])
elapsed_time = time.time() - start_time
print(elapsed_time)
hist_df = pd.DataFrame(history.history) 
# or save to csv: 
hist_csv_file = 'history_NoDecompostionRegression.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

#%%
# save trained weights
net_final.save(WEIGHTS_FINAL)
#net_final = load_model(WEIGHTS_FINAL)
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()

probabilities = net_final.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
prob_df = pd.DataFrame(probabilities) 
# or save to csv: 
prob_csv_file = 'prob_NoDecompositionRegression.csv'
with open(prob_csv_file, mode='w') as f:
    prob_df.to_csv(f)

'''
# PROB ANALYSIS
probabilities = pd.read_csv("prob_NoDecomposition.csv")
probabilities = probabilities.values
probabilities = probabilities[:, 1:]
from sklearn.metrics import confusion_matrix, classification_report

confidence = 0.90
#predicted_classes=np.argmax(probabilities,axis=1)
predicted_classes = np.where(probabilities > confidence)
aux_vector = np.arange(len(probabilities))
non_classified = list(set(aux_vector) - set(predicted_classes[0]))

true_classes = test_batches.classes
#true_classes = true_classes[predicted_classes[0]]
class_labels = list(test_batches.class_indices.keys())

predicted_classes_update = []
for i in range(len(probabilities)):
    
    if i in non_classified:
        predicted_classes_update.append(int(70))
    else:
        place = np.where(predicted_classes[0] == i)[0]
        predicted_classes_update.append(int(predicted_classes[1][place]))


#report = classification_report(true_classes, predicted_classes, target_names=class_labels)
#print(report)
#confusion_matrix = confusion_matrix(y_true=true_classes, y_pred=predicted_classes)
#print(confusion_matrix)

#report = classification_report(true_classes, predicted_classes[1], target_names=class_labels)
#print(report)
report = classification_report(true_classes, predicted_classes_update, target_names=class_labels)
print(report)



#confusion_matrix = confusion_matrix(y_true=true_classes, y_pred=predicted_classes[1])
confusion_matrix = confusion_matrix(y_true=true_classes, y_pred=predicted_classes_update)
print(confusion_matrix)

'''

'''
# training a c;assifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(probabilities, true_classes, test_size = 0.2, random_state=0)

clf = RandomForestClassifier(n_estimators=50, max_depth=7, oob_score = True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=class_labels))

from sklearn.dummy import DummyClassifier

clf_dummy = DummyClassifier().fit(X_train, y_train)
y_pred_dummy = clf_dummy.predict(X_test)
print(confusion_matrix(y_test, y_pred_dummy))
print(classification_report(y_test, y_pred_dummy, target_names=class_labels))
'''




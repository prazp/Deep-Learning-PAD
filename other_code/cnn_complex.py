#!/usr/bin/python3

import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import wavfile
import os
import pickle
import random
import librosa
import librosa.display
import time
import math
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import roc_curve, auc

NAME = "cnn_basic_speech-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

DATADIR = "/mnt/c/Users/prasa/code/Thesis"

#Training Data
pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_train.trn.pickle"), "rb")
trn_label_array = pickle.load(pickle_in)

#spoof is defined as 0 and genuine defined as 1
trn_label = []
for i in trn_label_array:
	if i[1] == b'spoof':
		trn_label.append(0)
	else:
		trn_label.append(1)

pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_train.pickle"), "rb")
trn_data = pickle.load(pickle_in)
melResultsTrain = []

for i in range(len(trn_data)):
	ps = librosa.feature.melspectrogram(trn_data[i].astype(float),sr=16000)
	if ps.shape != (128, 126): continue
	melResultsTrain.append((ps, trn_label[i]))

#Development Data
pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_dev.trl.pickle"), "rb")
trn_label_array = pickle.load(pickle_in)

#spoof is defined as 0 and genuine defined as 1
trn_label = []
for i in trn_label_array:
	if i[1] == b'spoof':
		trn_label.append(0)
	else:
		trn_label.append(1)

pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_dev.pickle"), "rb")
trn_data = pickle.load(pickle_in)
melResultsDev = []

for i in range(len(trn_data)):	
	ps = librosa.feature.melspectrogram(trn_data[i].astype(float),sr=16000)
	if ps.shape != (128, 126): continue
	melResultsDev.append((ps, trn_label[i]))

#need to do the following as I shuffle the data
random.shuffle(melResultsTrain)

#need to normalise the data

X_train = []
y_train = []
for res in melResultsTrain:
	X_train.append(res[0])
	y_train.append(res[1])

X_test = []
y_test = []
for res in melResultsDev:
	X_test.append(res[0])
	y_test.append(res[1])

print("Data Ready for Model")

# Reshape for CNN input
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

X_train = np.array([x.reshape( (128, 126, 1) ) for x in X_train])
X_test = np.array([x.reshape( (128, 126, 1) ) for x in X_test])

# One-Hot encoding for classes
y_train_saved = y_train
y_train = np.array(keras.utils.to_categorical(y_train, 2))
y_saved = y_test
y_test = np.array(keras.utils.to_categorical(y_test, 2))

#keras.normalise can be used

model = Sequential()
input_shape=(128, 126, 1)

model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Conv2D(32, (3, 3), padding="same"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Conv2D(32, (3, 3), padding="same"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(
	optimizer="Adam",
	loss="binary_crossentropy",
	metrics=['accuracy'])

model.fit(
	x=X_train, 
	y=y_train,
    epochs=4,
    batch_size=64,
    validation_data= (X_test, y_test), callbacks=[tensorboard])

score = model.evaluate(
	x=X_test,
	y=y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict(x=X_test)
print(predictions)

final = []

for pred in predictions:
	final.append(pred[1])

fpr, tpr, threshold = roc_curve(y_saved, final, pos_label=1)
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)
#!/usr/bin/python3
#Basic DNN CLassifer for PAD System

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

#Training data
pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_train.trn.pickle"), "rb")
trn_label_array = pickle.load(pickle_in)

#spoof is defined as 0 and genuine defined as 1
trn_label = []
for i in trn_label_array:
	if i[1] == b'spoof':
		trn_label.append(0)
	else:
		trn_label.append(1)

#Label data
pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_train.pickle"), "rb")
trn_data = pickle.load(pickle_in)
melResultsTrain = []

#extracting MFCC vectors from speech data
for i in range(len(trn_data)):
	ps = librosa.feature.mfcc(trn_data[i].astype(float)[0:32500], n_mfcc=20, sr=16000)
	if ps.shape != (20, 64): continue
	melResultsTrain.append((ps, trn_label[i]))

#Development Data
pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_dev.trl.pickle"), "rb")
dev_label_array = pickle.load(pickle_in)

#spoof is defined as 0 and genuine defined as 1
dev_label = []
for i in dev_label_array:
	if i[1] == b'spoof':
		dev_label.append(0)
	else:
		dev_label.append(1)

#Label data
pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_dev.pickle"), "rb")
dev_data = pickle.load(pickle_in)
melResultsDev = []

#extracting MFCC vectors from speech data
for i in range(len(dev_data)):	
	ps = librosa.feature.mfcc(dev_data[i].astype(float)[0:32500], n_mfcc=20, sr=16000)
	if ps.shape != (20, 64): continue
	melResultsDev.append((ps, dev_label[i]))

#randomise the data 
random.shuffle(melResultsTrain)

#extract training data and labels after shuffling
X_train = []
y_train = []
for res in melResultsTrain:
	X_train.append(res[0])
	y_train.append(res[1])

#extract evaluation data and labels after shuffling
X_test = []
y_test = []
for res in melResultsDev:
	X_test.append(res[0])
	y_test.append(res[1])

#need to normalise the data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

print("Data Ready for Model")

#reshape data so that it can be fed into DNN
X_train = np.array([x.reshape( (20, 64, 1) ) for x in X_train])
X_test = np.array([x.reshape( (20, 64, 1) ) for x in X_test])

#One-Hot encoding for two classes
y_train = np.array(keras.utils.to_categorical(y_train, 2))
y_saved = y_test
y_test = np.array(keras.utils.to_categorical(y_test, 2))

#neural network architecture
model = Sequential()

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy'])

#train neural network
model.fit(
	x=X_train, 
	y=y_train,
    epochs=10,
    batch_size=20,
    validation_data= (X_test, y_test), callbacks=[tensorboard])

#extract predicted posterior probabilities
predictions = model.predict(x=X_test)

#procedure to calculate EER
final = []
for pred in predictions:
	final.append(math.log(pred[1], 10))

fpr, tpr, threshold = roc_curve(y_saved, final, pos_label=1)
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)

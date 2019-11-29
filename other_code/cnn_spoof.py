#!/usr/bin/python3

import keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, TimeDistributed, LSTM, SimpleRNN, Masking, GRU, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import wavfile
import os
import pickle
import random
import time
import math
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import roc_curve, auc
from keras import backend as K
import resnet

NAME = "cnn-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

#DATADIR = "/mnt/c/Users/prasa/code/Thesis"
DATADIR = "../../../g/data1a/wa66/Prasanth"

#Training Data
pickle_in = open(os.path.join(DATADIR, "specTrain.pickle"), "rb")
specTrain = pickle.load(pickle_in)
pickle_in.close()

#Development Data
pickle_in = open(os.path.join(DATADIR, "specDev.pickle"), "rb")
specDev = pickle.load(pickle_in)
pickle_in.close()

X_train = []
y_train = []
for res in specTrain:
	X_train.append(res[0])
	y_train.append(res[1])

X_test = []
y_test = []
for res in specDev:
	X_test.append(res[0])
	y_test.append(res[1])
#need to normalise the data

print("Data Ready for Model")

#need to normalise the data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# Reshape for CNN input
X_train = np.array([x.reshape( (1025, 63, 1) ) for x in X_train])
X_test = np.array([x.reshape( (1025, 63, 1) ) for x in X_test])

# One-Hot encoding for classes
y_train_saved = y_train
y_train = np.array(keras.utils.to_categorical(y_train, 2))
y_test_saved = y_test
y_test = np.array(keras.utils.to_categorical(y_test, 2))

#keras.normalise can be used
# model = Sequential()
# input_shape=(1025, 63, 1)

# model.add(Conv2D(32, kernel_size=(3, 3), padding="same", input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(Conv2D(32, kernel_size=(3, 3), padding="same"))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2)))

# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2)))

# model.add(Flatten())
# model.add(Dense(64))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.5))

# model.add(Dense(64))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.6))

# model.add(Dense(64))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.7))

# model.add(Dense(2))
# model.add(Activation('softmax'))

model = Sequential()
input_shape=(1025, 63, 1)

model.add(Conv2D(32, kernel_size=(3, 3), padding="same", input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, kernel_size=(3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy'])

model.fit(
	x=X_train, 
	y=y_train,
    epochs=2,
    batch_size=128,
    validation_data= (X_test, y_test), callbacks=[tensorboard])


model.summary()

score = model.evaluate( x=X_train, y=y_train)

print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = model.evaluate( x=X_test, y=y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict(x=X_test)

final = []
f = open(os.path.join(DATADIR, "predictions.txt"), "w+")
for pred in predictions:
	f.write(str(pred[0]) + " " + str(pred[1]) + "\n")
	if pred[1] != 0:
		final.append(math.log(pred[1], 10))
	else:
		final.append(pred[1])

f.close()

f = open(os.path.join(DATADIR, "rnn_truth_train.txt"), "w+")
for value in y_train_saved:
	f.write(str(value) + "\n")
f.close()

f = open(os.path.join(DATADIR, "rnn_truth_test.txt"), "w+")
for value in y_test_saved:
	f.write(str(value) + "\n")
f.close()

fpr, tpr, threshold = roc_curve(y_test_saved, final, pos_label=1)
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)
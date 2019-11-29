#!/usr/bin/python3

import keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, TimeDistributed, Bidirectional, RNN, GRU, LSTM
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

NAME = "RNN-frontend-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

DATADIR = "/mnt/c/Users/prasa/code/Thesis"

#Training Data
pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_train.trn.pickle"), "rb")
trn_label_array = pickle.load(pickle_in)
pickle_in.close()

trn_label = []
for i in trn_label_array:
	if i[1] == b'spoof':
		trn_label.append(0)
	else:
		trn_label.append(1)

pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_train.pickle"), "rb")
trn_data = pickle.load(pickle_in)
pickle_in.close()
train_set = []

for i in range(len(trn_data)):
	j = 0
	ps = [] #len(dev_data[i])
	while j < 32000:
		ps.append(trn_data[i][j:j+320]);
		j += 320
	train_set.append((ps, trn_label[i]))

#Development Data
pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_dev.trl.pickle"), "rb")
dev_label_array = pickle.load(pickle_in)
pickle_in.close()

#spoof is defined as 0 and genuine defined as 1
dev_label = []
for i in dev_label_array:
	if i[1] == b'spoof':
		dev_label.append(0)
	else:
		dev_label.append(1)

pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_dev.pickle"), "rb")
dev_data = pickle.load(pickle_in)
pickle_in.close()
dev_set = []

for i in range(len(dev_data)):
	j = 0
	ps = [] #len(dev_data[i])
	while j < 32000:
		ps.append(dev_data[i][j:j+320]);
		j += 320
	dev_set.append((ps, dev_label[i]))

#need to do the following as I shuffle the data
random.shuffle(train_set)

X_train = []
y_train = []
for res in train_set:
	X_train.append(res[0])
	y_train.append(res[1])

X_test = []
y_test = []
for res in dev_set:
	X_test.append(res[0])
	y_test.append(res[1])

print("Data Ready for Model")

X_train = tf.keras.utils.normalize(X_train)
X_test = tf.keras.utils.normalize(X_test)

X_train = np.array(X_train)
X_test = np.array(X_test)

print(X_train.shape)
print(X_test.shape)

# One-Hot encoding for classes
y_train = np.array(keras.utils.to_categorical(y_train, 2))
y_saved = y_test
y_test = np.array(keras.utils.to_categorical(y_test, 2))

model = Sequential()
model.add(LSTM(100, input_shape=(100, 320), return_sequences = True))
model.add(LSTM(100))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy'])

print(model.summary())

# train LSTM
model.fit(
	x=X_train,
	y=y_train,
    epochs=3,
    batch_size=32, callbacks=[tensorboard])

# evaluate
score = model.evaluate( x=X_test, y=y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
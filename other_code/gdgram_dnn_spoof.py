#!/usr/bin/python3

import keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, TimeDistributed, LSTM, SimpleRNN, Masking, GRU, average
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
import os
import pickle
import random
import time
import math
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_curve, auc
from keras import backend as K
import h5py
import resnet

NAME = "group_delay_dnn_basic_speech-{}".format(int(time.time()))

early_stopper = EarlyStopping(min_delta=0.001, patience=10)
lr_reducer = ReduceLROnPlateau()
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

#DATADIR = "/mnt/c/Users/prasa/code/Thesis"
DATADIR = "../../../g/data1a/wa66/Prasanth"

#Training Data
pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_train.trn.pickle"), "rb")
trn_label_array = pickle.load(pickle_in)
pickle_in.close()

#spoof is defined as 0 and genuine defined as 1
trn_label = []
for i in trn_label_array:
	if i[1] == b'spoof':
		trn_label.append(0)
	else:
		trn_label.append(1)

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

with h5py.File(DATADIR+"/modified_group_delay.mat", 'r') as matFile:
	trainSet = np.transpose(np.array(matFile['trainSet']))
	devSet = np.transpose(np.array(matFile['devSet']))

print("Extracted data from mat file")

training = []
i = 0
for x in trainSet:
	training.append((x, trn_label[i]))
	i += 1

random.shuffle(training)

X_train = []
y_train = []

for res in training:
	X_train.append(res[0])
	y_train.append(res[1])

X_test = []
y_test = []
for x in devSet:
	X_test.append(x)

for y in dev_label:
	y_test.append(y)

print("Data Ready for Model")

#need to normalise the data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

X_train = np.array([x.reshape( (513, 256, 1) ) for x in X_train])
X_test = np.array([x.reshape( (513, 256, 1) ) for x in X_test])

# One-Hot encoding for classes
y_train_saved = y_train
y_train = np.array(keras.utils.to_categorical(y_train, 2))
y_test_saved = y_test
y_test = np.array(keras.utils.to_categorical(y_test, 2))

print(X_train.shape)
print(X_test.shape)

#keras.normalise can be used
model = resnet.ResnetBuilder.build_resnet_18((1, 513, 256), 2)

model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy'])

checkpoint_path=DATADIR+"/weights/weights_gdgram_spoof.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')

model.fit(
	x=X_train,
	y=y_train,
    epochs=150,
    batch_size=32,
    validation_data= (X_test, y_test), callbacks=[early_stopper, tensorboard, lr_reducer, checkpoint])

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

fpr, tpr, threshold = roc_curve(y_test_saved, final, pos_label=1)
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)
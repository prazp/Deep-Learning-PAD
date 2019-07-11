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

NAME = "rnn_basic_sines-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

DATADIR = "/mnt/c/Users/prasa/code/Thesis"

#DATADIR = "../../../g/data1a/wa66/Prasanth"

#Training Data
pickle_in = open(os.path.join(DATADIR, "genuine_sines.pickle"), "rb")
genuine_files = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(os.path.join(DATADIR, "spoof_sines.pickle"), "rb")
spoof_files = pickle.load(pickle_in)
pickle_in.close()

random.shuffle(genuine_files)
random.shuffle(spoof_files)

train_data = []
dev_data = []

for i in range(len(genuine_files)):
	j = 0
	ps = []
	while j < 32000:
		ps.append(genuine_files[i][j:j+320]);
		j += 320

	if i < 150:
		train_data.append((ps, 1))
	else:
		dev_data.append((ps, 1))

for i in range(len(spoof_files)):
	j = 0
	ps = []
	while j < 32000:
		ps.append(spoof_files[i][j:j+320]);
		j += 320

	if i < 150:
		train_data.append((ps, 0))
	else:
		dev_data.append((ps, 0))

#need to do the following as I shuffle the data
random.shuffle(train_data)

X_train = []
y_train = []
for res in train_data:
	X_train.append(res[0])
	y_train.append(res[1])

X_test = []
y_test = []
for res in dev_data:
	X_test.append(res[0])
	y_test.append(res[1])

#need to normalise the data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

print("Data Ready for Model")

X_train = np.array(X_train)
X_test = np.array(X_test)

# One-Hot encoding for classes
y_train_saved = y_train
y_train = np.array(keras.utils.to_categorical(y_train, 2))
y_test_saved = y_test
y_test = np.array(keras.utils.to_categorical(y_test, 2))

print(X_train.shape)
print(X_test.shape)

#keras.normalise can be used
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(10, 320)))
model.add(LSTM(24, return_sequences = True, dropout=0.5, recurrent_dropout=0.5, activation='relu'))
model.add(LSTM(24, name='dense_3', return_sequences = True, dropout=0.5, recurrent_dropout=0.5, activation='relu'))
model.add(TimeDistributed(Dense(2)))
model.add(Activation('sigmoid'))
# model.add(LSTM(64, return_sequences = True, activation='relu'))
# model.add(TimeDistributed(BatchNormalization()))
# model.add(LSTM(64, name='dense_3', return_sequences = True, activation='relu'))
# model.add(TimeDistributed(BatchNormalization()))
# model.add(TimeDistributed(Dense(2)))
# model.add(Activation('sigmoid'))

time_distributed_merge_layer = tf.keras.layers.Lambda(function=lambda x: tf.keras.backend.mean(x, axis=1))
model.add(time_distributed_merge_layer)

model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy'])

model.fit(
	x=X_train, 
	y=y_train,
    epochs=10,
    batch_size=128,
    validation_data= (X_test, y_test), callbacks=[tensorboard])

model.summary()

score = model.evaluate( x=X_test, y=y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict(x=X_test)

layer_name = "dense_3"
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

intermediate_output_train = intermediate_layer_model.predict(x=X_train)
intermediate_output_test = intermediate_layer_model.predict(x=X_test)

#get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[3].output])

# Testing
# test = np.random.random(input_shape)[np.newaxis,...]
# layer_outs = [func([test, 1.]) for func in functors]
# print layer_outs

print(intermediate_output_train.shape)
print(intermediate_output_test.shape)

final = []

pickle_out = open(os.path.join(DATADIR, "intermediate_output_train.pickle"), "wb")
pickle.dump(intermediate_output_train, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATADIR, "intermediate_output_test.pickle"), "wb")
pickle.dump(intermediate_output_test, pickle_out)
pickle_out.close()

f = open(os.path.join(DATADIR, "predictions.txt"), "w+")
for pred in predictions:
	f.write(str(pred[0]) + " " + str(pred[1]) + "\n")
	if pred[1] != 0:
		final.append(math.log(pred[1], 10))
	else:
		final.append(pred[1])

f.close()

f = open(os.path.join(DATADIR, "ground_truth_train.txt"), "w+")
for value in y_train_saved:
	f.write(str(value) + "\n")
f.close()

f = open(os.path.join(DATADIR, "ground_truth_test.txt"), "w+")
for value in y_test_saved:
	f.write(str(value) + "\n")
f.close()

fpr, tpr, threshold = roc_curve(y_test_saved, final, pos_label=1)
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)

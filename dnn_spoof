#!/usr/bin/python3

import keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, TimeDistributed, LSTM, SimpleRNN, Masking, GRU
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import wavfile
import os
import pickle
import random
#import librosa
import time
import math
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import roc_curve, auc
from keras import backend as K

NAME = "dnn_basic_speech-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

DATADIR = "/mnt/c/Users/prasa/code/Thesis"

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

pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_train.pickle"), "rb")
trn_data = pickle.load(pickle_in)
pickle_in.close()
melResultsTrain = []

for i in range(len(trn_data)):
	j = 0
	ps = [] 
	while j+160 < len(trn_data[i]):
		ps.append(trn_data[i][j:j+320]);
		j += 160
	melResultsTrain.append((ps, trn_label[i]))

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
melResultsDev = []

for i in range(len(dev_data)):	
	j = 0
	ps = [] 
	while j+160 < len(dev_data[i]):
		ps.append(dev_data[i][j:j+320]);
		j += 160
	melResultsDev.append((ps, dev_label[i]))

#need to do the following as I shuffle the data
random.shuffle(melResultsTrain)

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

#need to normalise the data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

print("Data Ready for Model")

#X_train = np.array([x.reshape( (20, 126, 1) ) for x in X_train])
#X_test = np.array([x.reshape( (20, 126, 1) ) for x in X_test])

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

model.add(Masking(mask_value=0.0, input_shape=(199, 400)))
model.add(TimeDistributed(Dense(128)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(TimeDistributed(Dense(128)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(TimeDistributed(Dense(128), name='dense_3'))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(TimeDistributed(Dense(2)))
model.add(Activation('sigmoid'))

time_distributed_merge_layer = tf.keras.layers.Lambda(function=lambda x: tf.keras.backend.mean(x, axis=1))
model.add(time_distributed_merge_layer)

model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy'])


print(y_train)
print(y_test)

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

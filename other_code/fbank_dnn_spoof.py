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
import librosa
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import roc_curve, auc
from keras import backend as K

NAME = "fbank_dnn_basic_speech-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

DATADIR = "/mnt/c/Users/prasa/code/Thesis"

#Training Data
pickle_in = open(os.path.join(DATADIR, "fbankTrain.pickle"), "rb")
fbankTrain = pickle.load(pickle_in)
pickle_in.close()

#Development Data
pickle_in = open(os.path.join(DATADIR, "fbankDev.pickle"), "rb")
fbankDev = pickle.load(pickle_in)
pickle_in.close()

trainSet = []
y_train = []
for res in fbankTrain:
	trainSet.append(res[0])
	y_train.append(res[1])

testSet = []
y_test = []
for res in fbankDev:
	testSet.append(res[0])
	y_test.append(res[1])

X_train = []
for x in trainSet:
	delta = librosa.feature.delta(x)
	delta2 = librosa.feature.delta(x, order=2)
	temp = np.concatenate((x,delta,delta2),axis=0)
	X_train.append(temp)

X_test = []
for x in testSet:
	delta = librosa.feature.delta(x)
	delta2 = librosa.feature.delta(x, order=2)
	temp = np.concatenate((x,delta,delta2),axis=0)
	X_test.append(temp)

print("Data Ready for Model")

#need to normalise the data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

X_train = np.array([x.reshape( (99*3, 52) ) for x in X_train])
X_test = np.array([x.reshape( (99*3, 52) ) for x in X_test])

# One-Hot encoding for classes
y_train_saved = y_train
y_train = np.array(keras.utils.to_categorical(y_train, 2))
y_test_saved = y_test
y_test = np.array(keras.utils.to_categorical(y_test, 2))

print(X_train.shape)
print(X_test.shape)

#keras.normalise can be used
model = Sequential()

# model.add(Flatten(input_shape=(1, 99*52)))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.5))

# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.5))

# model.add(Dense(256, name='dense_3'))
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.5))

# model.add(Dense(2))
# model.add(Activation('sigmoid'))

#model.add(Masking(mask_value=0.0, input_shape=(73, 6570)))
# model.add(TimeDistributed(Dense(256), input_shape=(99*3, 52)))
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.5))

# model.add(TimeDistributed(Dense(256)))
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.5))

# model.add(TimeDistributed(Dense(256), name='dense_3'))
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.5))

# model.add(TimeDistributed(Dense(2)))
# model.add(Activation('sigmoid'))

model.add(LSTM(128, return_sequences = True, dropout=0.2, recurrent_dropout=0.2, activation='relu', input_shape=(99*3, 52)))
model.add(LSTM(128, name='dense_3', return_sequences = True, dropout=0.2, recurrent_dropout=0.2, activation='relu'))
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
    epochs=5,
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
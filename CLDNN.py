#!/usr/bin/python3

import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Lambda, Input, Dense, Dropout, Activation, Flatten, Conv1D, Conv2D, ZeroPadding2D, MaxPooling2D, BatchNormalization, UpSampling2D, GlobalMaxPooling1D, Add, Multiply, multiply, SpatialDropout2D, Reshape, GlobalAveragePooling2D, TimeDistributed, LSTM, SimpleRNN, Masking, Permute, Reshape
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import wavfile
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import os
import pickle
import random
import time
import math
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, EarlyStopping, Callback
from sklearn.metrics import roc_curve, auc
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
import h5py
from sklearn import preprocessing
import speechpy

import resnet

NAME = "rnn_basic_speech-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

#DATADIR = "/mnt/c/Users/prasa/code/Thesis"
DATADIR = "../../../g/data1a/wa66/Prasanth"
bestEER = 1

#callback run every epoch
class developmentSetEval(Callback):

    def __init__(self, y_test_saved):
        super(developmentSetEval, self).__init__()
        self.y_test_saved = y_test_saved

    def on_epoch_end(self, epoch, logs=None):
        x_test = self.validation_data[0]
        y_test = self.validation_data[1]

        # score = self.model.evaluate(x=x_test, y=y_test, verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])
        
        predictions = self.model.predict(x=x_test)

        final = []
        for pred in predictions:
            final.append(pred[1])
        
        fpr, tpr, threshold = roc_curve(self.y_test_saved, final, pos_label=1)
        fnr = 1 - tpr
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        print(EER)

        global bestEER
        if EER < bestEER:
            #model.save_weights(DATADIR+"/weights/"+NAME+".hdf5")
            #model.save(DATADIR+"/weights/attention"+str(EER)+".hdf5")
            bestEER = EER

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

size = 80000
hann_window = np.hanning(400).reshape(-1,1)

for i in range(len(trn_data)):
	j = 0
	ps = []

	if (len(trn_data[i]) < size):
		trn_data[i] = np.pad(trn_data[i],(0,size-len(trn_data[i])),'wrap')

	trn_data[i] = trn_data[i][0:size].reshape(-1,1)
	trn_data[i] = speechpy.processing.cmvn(trn_data[i],variance_normalization=True)

	while j < size-400:
		ps.append(trn_data[i][j:j+400]*hann_window)
		j += 160
	
	melResultsTrain.append((ps, trn_label[i]))

del trn_data
print("Done Training")

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

	if (len(dev_data[i]) < size):
		dev_data[i] = np.pad(dev_data[i],(0,size-len(dev_data[i])),'wrap')

	dev_data[i] = dev_data[i][0:size].reshape(-1,1)
	dev_data[i] = speechpy.processing.cmvn(dev_data[i],variance_normalization=True)

	#dev_data[i] = StandardScaler().fit_transform(dev_data[i][0:174400].reshape(-1,1))

	while j < size-400:
		ps.append(dev_data[i][j:j+400]*hann_window)
		j += 160

	melResultsDev.append((ps, dev_label[i]))

del dev_data
print("Done Development")

#Evaluation Data
pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_eval.trl.pickle"), "rb")
eval_label_array = pickle.load(pickle_in)
pickle_in.close()

#spoof is defined as 0 and genuine defined as 1
eval_label = []
for i in eval_label_array:
	if i[1] == b'spoof':
		eval_label.append(0)
	else:
		eval_label.append(1)

pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_eval.pickle"), "rb")
eval_data = pickle.load(pickle_in)
pickle_in.close()
melResultsEval = []

for i in range(len(eval_data)):
	j = 0
	ps = []

	if (len(eval_data[i]) < size):
		eval_data[i] = np.pad(eval_data[i],(0,size-len(eval_data[i])),'wrap')

	eval_data[i] = eval_data[i][0:size].reshape(-1,1)
	eval_data[i] = speechpy.processing.cmvn(eval_data[i],variance_normalization=True)

	#eval_data[i] = StandardScaler().fit_transform(eval_data[i][0:174400].reshape(-1,1))

	while j < size-400:
		ps.append(eval_data[i][j:j+400]*hann_window)
		j += 160

	melResultsEval.append((ps, eval_label[i]))

del eval_data
print("Done Evaluation")

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

X_eval = []
y_eval = []
for res in melResultsEval:
	X_eval.append(res[0])
	y_eval.append(res[1])

print("Data Ready for Model")

X_train = np.array(X_train)
X_test = np.array(X_test)
X_eval = np.array(X_eval)

print(X_train.shape)
print(X_test.shape)
print(X_eval.shape)

X_train = np.array([x.reshape( (498, 20, 20, 1) ) for x in X_train])
X_test = np.array([x.reshape( (498, 20, 20, 1) ) for x in X_test])
X_eval = np.array([x.reshape( (498, 20, 20, 1) ) for x in X_eval])

# One-Hot encoding for classes
y_train_saved = y_train
y_train = np.array(tf.keras.utils.to_categorical(y_train, 2))
y_test_saved = y_test
y_test = np.array(tf.keras.utils.to_categorical(y_test, 2))
y_eval_saved = y_eval
y_eval = np.array(tf.keras.utils.to_categorical(y_eval, 2))

print(X_train.shape)
print(X_test.shape)
print(X_eval.shape)

lr_reducer = ReduceLROnPlateau(patience=1,factor=0.5)
early_stopper = EarlyStopping(min_delta=0.001, patience=5)

#setting seed for reporducability
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)
tf.reset_default_graph()

#keras.normalise can be used
model = Sequential()

# model.add(TimeDistributed(Conv1D(40, kernel_size=3, strides=1),input_shape=(498, 400, 1)))
# model.add(TimeDistributed(BatchNormalization()))
# model.add(TimeDistributed(Activation('relu')))
# model.add(TimeDistributed(GlobalMaxPooling1D()))

# model.add(TimeDistributed(Reshape((40,1))))
# model.add(TimeDistributed(Conv1D(256, kernel_size=3, strides=1)))
# model.add(TimeDistributed(BatchNormalization()))
# model.add(TimeDistributed(Activation('relu')))

# model.add(TimeDistributed(GlobalMaxPooling1D()))
# # model.add(TimeDistributed(Flatten()))
# # model.add(TimeDistributed(Dense(256)))
# # model.add(TimeDistributed(Dropout(0.5)))
# # model.add(TimeDistributed(Dense(2)))
# # model.add(TimeDistributed(Activation('sigmoid')))

# # model.add(LSTM(64, return_sequences = True, dropout=0.2, recurrent_dropout=0.2, activation='relu'))
# model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, activation='relu'))
# model.add(Dense(512))
# model.add(Dropout(0.5))
# model.add(Dense(2))
# model.add(Activation('sigmoid'))

#CLDNN
model.add(TimeDistributed(Conv2D(16, kernel_size=7, strides=2, padding='same'),input_shape=(498, 20, 20, 1)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation("relu")))

model.add(TimeDistributed(Conv2D(32, kernel_size=5, strides=2, padding='same')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation("relu")))

model.add(TimeDistributed(Conv2D(64, kernel_size=3, strides=2, padding='same')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation("relu")))

model.add(TimeDistributed(Reshape((64,9))))
model.add(TimeDistributed(Flatten()))

model.add(TimeDistributed(LSTM(128, return_sequences=True)))
model.add(TimeDistributed(LSTM(128))) #should mean across all the vectors using lamdba function

model.add(TimeDistributed(Dense(64)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation("relu")))

model.add(TimeDistributed(Dense(128), name='dense_inter'))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation("relu")))

model.add(TimeDistributed(Dense(2)))
model.add(TimeDistributed(Activation('sigmoid')))

time_distributed_merge_layer = Lambda(function=lambda x: tf.keras.backend.mean(x, axis=1))
model.add(time_distributed_merge_layer)

model.add(LSTM(128, return_sequences=True, name='dense_inter'))
model.add(LSTM(128)) #should mean across all the vectors using lamdba function

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(2))
model.add(Activation('sigmoid'))

model.summary()

adam = optimizers.Adam(amsgrad=False)

model.compile(
	optimizer=adam,
	loss="categorical_crossentropy",
	metrics=['accuracy'])

model.fit(
	x=X_train,
	y=y_train,
    epochs=30,
    batch_size=16,
    verbose=2,
    validation_data= (X_test, y_test), callbacks=[early_stopper, lr_reducer, tensorboard, developmentSetEval(y_test_saved)])

score = model.evaluate( x=X_train, y=y_train, verbose=0)

print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = model.evaluate( x=X_test, y=y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

score = model.evaluate( x=X_eval, y=y_eval, verbose=0)

print('Eval loss:', score[0])
print('Eval accuracy:', score[1])

predictions = model.predict(x=X_eval)

final = []
f = open(os.path.join(DATADIR, "predictions.txt"), "w+")
for pred in predictions:
	f.write(str(pred[0]) + " " + str(pred[1]) + "\n")
	final.append(pred[1])

f.close()

fpr, tpr, threshold = roc_curve(y_eval_saved, final, pos_label=1)
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)

#NEXT STAGE - to ResNet
layer_name = "dense_inter"
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

intermediate_output_train = intermediate_layer_model.predict(x=X_train)
intermediate_output_test = intermediate_layer_model.predict(x=X_test)
intermediate_output_eval = intermediate_layer_model.predict(x=X_eval)

print(intermediate_output_train.shape)
print(intermediate_output_test.shape)
print(intermediate_output_eval.shape)

X_train = []
for x in intermediate_output_train:
    #x = preprocessing.StandardScaler().fit_transform(np.transpose(x))
    x = speechpy.processing.cmvn(np.transpose(x),variance_normalization=True)
    X_train.append(np.transpose(x))

X_test = []
for x in intermediate_output_test:
    #x = preprocessing.StandardScaler().fit_transform(np.transpose(x))
    x = speechpy.processing.cmvn(np.transpose(x),variance_normalization=True)
    X_test.append(np.transpose(x))

X_eval = []
for x in intermediate_output_eval:
    #x = preprocessing.StandardScaler().fit_transform(np.transpose(x))
    x = speechpy.processing.cmvn(np.transpose(x),variance_normalization=True)
    X_eval.append(np.transpose(x))

X_train = np.array([x.reshape( (128, 498, 1) ) for x in X_train])
X_test = np.array([x.reshape( (128, 498, 1) ) for x in X_test])
X_eval = np.array([x.reshape( (128, 498, 1) ) for x in X_eval])

print("Data Ready for Model")

model = resnet.ResnetBuilder.build_resnet_18((1, 128, 498), 2)

model.summary()

adam = optimizers.Adam(amsgrad=False)

model.compile(
        optimizer=adam,
        loss="categorical_crossentropy",
        metrics=['accuracy'])

model.fit(
        x=X_train,
        y=y_train,
        epochs=30,
        batch_size=8,
        verbose=2,
        validation_data= (X_test, y_test), callbacks=[early_stopper, lr_reducer, tensorboard, developmentSetEval(y_test_saved)])

score = model.evaluate( x=X_train, y=y_train, verbose=0)

print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = model.evaluate( x=X_test, y=y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

##validation data
print("Validation of final model")
score = model.evaluate( x=X_eval, y=y_eval, verbose=0)

print('Eval loss:', score[0])
print('Eval accuracy:', score[1])

predictions = model.predict(x=X_eval)

final = []
for pred in predictions:
    final.append(pred[1])

fpr, tpr, threshold = roc_curve(y_eval_saved, final, pos_label=1)
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)

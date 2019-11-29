#!/usr/bin/python3

import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Lambda, Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, TimeDistributed, LSTM, SimpleRNN, Masking, GRU, BatchNormalization
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

import senet

#DATADIR = "/mnt/c/Users/prasa/code/Thesis"
DATADIR = "../../../g/data1a/wa66/Prasanth"
bestEER = 1

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

        print("Alternative EER")
        EER = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1)
        print(EER)

        global bestEER
        if EER < bestEER:
            model.save_weights(DATADIR+"/weights/"+NAME+".hdf5")
            bestEER = EER


NAME = "senet-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
lr_reducer = ReduceLROnPlateau(patience=1,factor=0.5)
early_stopper = EarlyStopping(min_delta=0.001, patience=5)

#setting seed for reporducability
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)
tf.reset_default_graph()

#Training Data
pickle_in = open(os.path.join(DATADIR, "specTrain_win400_hop160_cmvn.pickle"), "rb")
specTrain = pickle.load(pickle_in)
pickle_in.close()

#Development Data
pickle_in = open(os.path.join(DATADIR, "specDev_win400_hop160_cmvn.pickle"), "rb")
specDev = pickle.load(pickle_in)
pickle_in.close()

#Evaluation Data
pickle_in = open(os.path.join(DATADIR, "specEval_win400_hop160_cmvn.pickle"), "rb")
specEval = pickle.load(pickle_in)
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

X_eval = []
y_eval = []
for res in specEval:
    X_eval.append(res[0])
    y_eval.append(res[1])

print("Data Ready for Model")

#need to normalise the data
# X_train = tf.keras.utils.normalize(X_train, axis=1)
# X_test = tf.keras.utils.normalize(X_test, axis=1)
# X_eval = tf.keras.utils.normalize(X_eval, axis=1)

# Reshape for CNN input
X_train = np.array([x.reshape( (257, 1091, 1) ) for x in X_train])
X_test = np.array([x.reshape( (257, 1091, 1) ) for x in X_test])
X_eval = np.array([x.reshape( (257, 1091, 1) ) for x in X_eval])

# One-Hot encoding for classes
y_train_saved = y_train
y_train = np.array(tf.keras.utils.to_categorical(y_train, 2))
y_test_saved = y_test
y_test = np.array(tf.keras.utils.to_categorical(y_test, 2))
y_eval_saved = y_eval
y_eval = np.array(tf.keras.utils.to_categorical(y_eval, 2))

print("Data Ready for Model")

model = senet.SenetBuilder.build_senet_18((1, 257, 1091), 2)

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

#validation data
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

print("Validation of best model!!!!")
del model
model = senet.SenetBuilder.build_senet_18((1, 257, 1091), 2)
model.load_weights(DATADIR+"/weights/"+NAME+".hdf5")

model.compile(
        optimizer=adam,
        loss="categorical_crossentropy",
        metrics=['accuracy'])

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

# EER reference: https://yangcha.github.io/EER-ROC/
print("Alternative EER")
EER = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1)
print(EER)
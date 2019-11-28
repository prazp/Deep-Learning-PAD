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
from tensorflow.python.keras import losses

import resnet
from losses import categorical_focal_loss

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

        global model_class
        
        predictions = model_class.predict(x=x_test)

        final = []
        for pred in predictions:
            final.append(pred[1])
        
        fpr, tpr, threshold = roc_curve(self.y_test_saved, final, pos_label=1)
        fnr = 1 - tpr
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        print(EER)
 
        print("Second EER measure")
        EER = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1)
        print(EER)

        global bestEER
        if EER < bestEER:
            model_class.save(DATADIR+"/weights/"+NAME+".hdf5")
            #model_class.save_weights(DATADIR+"/weights/"+NAME+".hdf5")
            bestEER = EER

#loss function only calculates loss on source domain
def masked_loss_function(y_true, y_pred):
    # mask = K.cast(K.not_equal(y_true, [0, 0]), K.floatx())
    # loss = K.categorical_crossentropy(y_true * mask, y_pred * mask)
    # return K.switch(K.flatten(K.equal(y_true, [0, 0])), K.zeros_like(loss), loss)

    idx  = tf.not_equal(y_true, [-1,-1])
    y_true = tf.boolean_mask(y_true, idx)
    y_pred = tf.boolean_mask(y_pred, idx)
    return losses.categorical_crossentropy(y_true, y_pred)

class updateVariable(Callback):
    def __init__(self, alpha):
        self.alpha = alpha       
    def on_epoch_begin(self, epoch, logs={}):
        temp_lambda = 2/(1+np.exp(-0.001*epoch))-1
        print(temp_lambda)
        K.set_value(self.alpha, temp_lambda)

NAME = "resnet_advesarial-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
lr_reducer = ReduceLROnPlateau(patience=2,factor=0.5, verbose=0)
early_stopper = EarlyStopping(min_delta=0.001, patience=5)

#setting seed for reporducability
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)
tf.reset_default_graph()

#source domain: ASVspoof 2017 trainset, target domain: ASVspoof 2017 evalset

#Training Data
pickle_in = open(os.path.join(DATADIR, "specTrain_win400_hop160_standard_correct.pickle"), "rb")
specTrain = pickle.load(pickle_in)
pickle_in.close()

#Development Data
pickle_in = open(os.path.join(DATADIR, "specDev_win400_hop160_standard_correct.pickle"), "rb")
specDev = pickle.load(pickle_in)
pickle_in.close()

#Evaluation Data
pickle_in = open(os.path.join(DATADIR, "specEval_win400_hop160_standard_correct.pickle"), "rb")
specEval = pickle.load(pickle_in)
pickle_in.close()

X_domain = []
y_domain = [] #0 means train set and 1 means dev set
y_class = []

X_train = []
y_train = []
for res in specTrain:
    X_train.append(res[0])
    y_train.append(res[1])
    X_domain.append(res[0])
    y_domain.append(0)
    y_class.append(res[1])

X_test = []
y_test = []
for res in specDev:
    X_test.append(res[0])
    y_test.append(res[1])

X_eval = []
y_eval = []
i = 0
for res in specEval:
    X_eval.append(res[0])
    y_eval.append(res[1])
    #if i < 6028:
    X_domain.append(res[0])
    y_domain.append(1)
    #y_class.append(res[1])
    #i += 1

print("Data Ready for Model")

# Reshape for CNN input
X_train = np.array([x.reshape( (257, 1091, 1) ) for x in X_train])
X_test = np.array([x.reshape( (257, 1091, 1) ) for x in X_test])
X_eval = np.array([x.reshape( (257, 1091, 1) ) for x in X_eval])
X_domain = np.array([x.reshape( (257, 1091, 1) ) for x in X_domain])

# One-Hot encoding for classes
y_train_saved = y_train
y_train = np.array(tf.keras.utils.to_categorical(y_train, 2))
y_test_saved = y_test
y_test = np.array(tf.keras.utils.to_categorical(y_test, 2))
y_eval_saved = y_eval
y_eval = np.array(tf.keras.utils.to_categorical(y_eval, 2))
y_domain_saved = y_domain
y_domain = np.array(tf.keras.utils.to_categorical(y_domain, 2))
y_class_saved = y_class
y_class = np.array(tf.keras.utils.to_categorical(y_class, 2))

i = 0
while i < 13306:
    y_class = np.append(y_class, [[-1, -1]],axis=0)
    i += 1

y_test_domain = []
i = 0
while i < 1710:
    y_test_domain.append(1)
    i += 1

y_test_domain = np.array(tf.keras.utils.to_categorical(y_test_domain, 2))

print(y_test_domain.shape)

print("Data Ready for Model")

hp_lambda = K.variable(1.)

model_combined, model_class = resnet.DomainResnetBuilder.build_resnet_18((1, 257, 1091), 2, hp_lambda)

# model_class.summary()
# model_domain.summary()

adam = optimizers.Adam(amsgrad=False)
sgd = optimizers.SGD(lr=0.001, momentum=0.9)

model_combined.compile(optimizer=sgd, loss=[masked_loss_function, categorical_focal_loss(alpha=.25, gamma=5)], metrics=["accuracy"])

# model_combined.compile(
#         optimizer=adam,
#         loss=[masked_loss_function, "categorical_crossentropy"],
#         metrics=['accuracy'])

model_combined.fit(
        x=X_domain,
        y=[y_class, y_domain],
        epochs=10,
        batch_size=8,
        verbose=2,
        validation_data= (X_test, [y_test, y_test_domain]), callbacks=[lr_reducer, tensorboard, developmentSetEval(y_test_saved)])

#early_stopper, updateVariable(hp_lambda)

model_class.compile(
        optimizer=sgd,
        loss=[masked_loss_function],
        metrics=['accuracy'])

score = model_class.evaluate( x=X_train, y=y_train, verbose=0)

print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = model_class.evaluate( x=X_test, y=y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

##validation data
print("Validation of final model")
score = model_class.evaluate( x=X_eval, y=y_eval, verbose=0)

print('Eval loss:', score[0])
print('Eval accuracy:', score[1])

predictions = model_class.predict(x=X_eval)

final = []
for pred in predictions:
    final.append(pred[1])

fpr, tpr, threshold = roc_curve(y_eval_saved, final, pos_label=1)
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)

print("Validation of best model!!!!")
del model_class, model_combined
# model_combined, model_domain = resnet.DomainResnetBuilder.build_resnet_18((1, 257, 1091), 2)
# model_class.load_weights(DATADIR+"/weights/"+NAME+".hdf5")
model_class = load_model(DATADIR+"/weights/"+NAME+".hdf5")

model_class.compile(
        optimizer=sgd,
        loss="categorical_crossentropy",
        metrics=['accuracy'])

score = model_class.evaluate( x=X_eval, y=y_eval, verbose=0)

print('Eval loss:', score[0])
print('Eval accuracy:', score[1])

predictions = model_class.predict(x=X_eval)

final = []
for pred in predictions:
    final.append(pred[1])

fpr, tpr, threshold = roc_curve(y_eval_saved, final, pos_label=1)
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)

# EER reference: https://yangcha.github.io/EER-ROC/
EER = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1)
print(EER)

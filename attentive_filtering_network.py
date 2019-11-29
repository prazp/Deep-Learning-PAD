#!/usr/bin/python3

#Use_bias is set back to True, glorot_normal

import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Lambda, Input, Dense, Dropout, Activation, Flatten, Conv2D, ZeroPadding2D, MaxPooling2D, BatchNormalization, UpSampling2D, Add, Multiply, multiply, SpatialDropout2D, Reshape, GlobalAveragePooling2D
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
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, Callback
from sklearn.metrics import roc_curve, auc
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
import h5py

import losses

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
            final.append(pred[0])
        
        fpr, tpr, threshold = roc_curve(self.y_test_saved, final, pos_label=1)
        fnr = 1 - tpr
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        print(EER)

        print("Second EER measure")
        EER = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1)
        print(EER)

        global bestEER
        if EER < bestEER:
            model.save_weights(DATADIR+"/weights/"+NAME+".hdf5")
            #model.save(DATADIR+"/weights/"+NAME+".hdf5")
            bestEER = EER

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=3)(input)
    return Activation("relu")(norm)

def _bn_relu_dense(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=1)(input)
    return Activation("relu")(norm)

def resize_like(input_tensor, ref_tensor): # resizes input tensor wrt. ref_tensor
    H, W = ref_tensor[0], ref_tensor[1]
    return tf.image.resize_images(input_tensor, [H, W])

def residualBlock(**conv_params):
    """residual block
    |-->bn2d-->relu-->conv2d-->bn2d-->relu-->conv2d--|
    x -----------------------------------------------+-->maxpool-->dilated_conv2d-->out
    """
    pool_size = conv_params.setdefault("pool_size", (1, 1))
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "glorot_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    dilation_rate = conv_params.setdefault("dilation_rate", (1,1))

    def f(input):
        x = _bn_relu(input)
        x = Conv2D(32, kernel_size=(3, 3), strides=strides, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = _bn_relu(x)
        x = Conv2D(32, kernel_size=(3, 3), strides=strides, padding=padding, kernel_initializer=kernel_initializer)(x)
        #x = squeeze_excite_block()(x) 
        x = Add()([x, input]) #x += input
        x = MaxPooling2D(pool_size=pool_size)(x) #(vertical, horizontal)
        x = Conv2D(32, kernel_size=(3, 3), dilation_rate=dilation_rate, kernel_initializer=kernel_initializer)(x)
        return x

    return f

#constructs the attention network
def attentionModule(input):

    residual = input

    x = Conv2D(16, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer="glorot_normal")(input)
    x = _bn_relu(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer="glorot_normal")(x)
    x = _bn_relu(x)

    ## softmax branch: bottom-up 
    x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)
    x = Conv2D(16, kernel_size=(3, 3), dilation_rate=(4,8), kernel_initializer="glorot_normal")(x)
    x = ZeroPadding2D()(x)
    x = _bn_relu(x)

    out_skip1 = Conv2D(16, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer="glorot_normal")(x)
    out_skip1 = _bn_relu(out_skip1)
    print(out_skip1.shape)

    x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)
    x = Conv2D(16, kernel_size=(3, 3), dilation_rate=(8,16), kernel_initializer="glorot_normal")(x)
    x = ZeroPadding2D()(x)
    x = _bn_relu(x)

    out_skip2 = Conv2D(16, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer="glorot_normal")(x)
    out_skip2 = _bn_relu(out_skip2)
    print(out_skip2.shape)

    x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)
    x = Conv2D(16, kernel_size=(3, 3), dilation_rate=(16,32), kernel_initializer="glorot_normal")(x)
    x = ZeroPadding2D()(x)
    x = _bn_relu(x)

    out_skip3 = Conv2D(16, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer="glorot_normal")(x)
    out_skip3 = _bn_relu(out_skip3)
    print(out_skip3.shape)

    x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)
    x = Conv2D(16, kernel_size=(3, 3), dilation_rate=(32,64), kernel_initializer="glorot_normal")(x)
    x = ZeroPadding2D()(x)
    x = _bn_relu(x)

    out_skip4 = Conv2D(16, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer="glorot_normal")(x)
    out_skip4 = _bn_relu(out_skip4)
    print(out_skip4.shape)
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(1, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), dilation_rate=(64,128), kernel_initializer="glorot_normal")(x)
    x = ZeroPadding2D()(x)
    x = _bn_relu(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="glorot_normal")(x)
    x = _bn_relu(x)
    
    #top down
    print(x.shape)
    #x = UpSamplingUnet(size=(905/454,1), interpolation='bilinear')(x)

    x = Lambda(resize_like, arguments={'ref_tensor':(137,851)})(x)
    x = Add()([x, out_skip4]) #x += out_skip4
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="glorot_normal")(x)
    x = _bn_relu(x)

    #x = UpSamplingUnet(size=((969/907),37/5), interpolation='bilinear')(x)
    x = Lambda(resize_like, arguments={'ref_tensor':(201,979)})(x)
    x = Add()([x, out_skip3]) #x += out_skip3
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="glorot_normal")(x)
    x = _bn_relu(x)
    
    #x = UpSampling2D(size=(1,1), interpolation='bilinear')(x)
    x = Lambda(resize_like, arguments={'ref_tensor':(233,1043)})(x)
    x = Add()([x, out_skip2]) #x += out_skip2
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="glorot_normal")(x)
    x = _bn_relu(x)  
    
    #x = UpSampling2D(size=(1,1), interpolation='bilinear')(x)
    x = Lambda(resize_like, arguments={'ref_tensor':(249,1075)})(x)
    x = Add()([x, out_skip1]) #x += out_skip1
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="glorot_normal")(x)
    x = _bn_relu(x)
    
    #x = UpSampling2D(size=(1,1), interpolation='bilinear')(x)
    x = Lambda(resize_like, arguments={'ref_tensor':(257,1091)})(x)

    # x = _bn_relu(x)
    # x = Conv2D(4, kernel_size=(1, 1), strides=(1,1), padding="same", kernel_initializer="glorot_normal")(x)
    # x = _bn_relu(x)
    # x = Conv2D(1, kernel_size=(1, 1), strides=(1,1), padding="same", kernel_initializer="glorot_normal")(x)

    x = _bn_relu(x)
    x = Conv2D(32, kernel_size=(1, 1), strides=(1,1), padding="same", use_bias=False)(x)
    x = _bn_relu(x)
    x = Conv2D(32, kernel_size=(1, 1), strides=(1,1), padding="same", use_bias=False)(x)


    weight = Activation('sigmoid')(x) #output of softmax layer is weight

    x = Multiply()([weight, residual])

    x = Add()([x, residual])

    return x, weight

def build_model():
    input = Input(shape=(257, 1091, 1))
    [x, weight] = attentionModule(input)

    x = Conv2D(16, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer="glorot_normal")(x)

    #block 1
    residual = x
    x = _bn_relu(x)
    x = Conv2D(16, kernel_size=(3,   3), strides=(1,1), padding="same", kernel_initializer="glorot_normal")(x)
    x = _bn_relu(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer="glorot_normal")(x)
    x = Add()([x, residual]) #x += residual
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), dilation_rate=(2,2), kernel_initializer="glorot_normal")(x)
    x = ZeroPadding2D()(x)

    x = residualBlock(pool_size=(1,2), dilation_rate=(4,4))(x)
    x = residualBlock(pool_size=(2,2), dilation_rate=(4,4))(x)
    x = residualBlock(pool_size=(2,2), dilation_rate=(8,8))(x)
    x = residualBlock(pool_size=(2,2), dilation_rate=(8,8))(x)

    x = Flatten()(x)

    x = Dense(32, kernel_initializer="glorot_normal")(x)

    residual = x

    x = _bn_relu_dense(x)
    x = Dense(32, kernel_initializer="glorot_normal")(x)
    x = _bn_relu_dense(x)
    x = Dense(32, kernel_initializer="glorot_normal")(x)
    x = Add()([x, residual]) #x += residual
    ###

    residual = x
    x = _bn_relu_dense(x)
    x = Dense(32, kernel_initializer="glorot_normal")(x)
    x = _bn_relu_dense(x)
    x = Dense(32, kernel_initializer="glorot_normal")(x)
    x = Add()([x, residual]) #x += residual

    ###
    x = _bn_relu_dense(x)
    dense = Dense(1, kernel_initializer="glorot_normal", activation="sigmoid")(x)

    model = Model(inputs=input, outputs=dense)
    weight_model = Model(inputs=input, outputs=weight)
    return model, weight_model

NAME = "attention-{}".format(int(time.time()))

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
#need to normalise the data

print("Data Ready for Model")

# Reshape for CNN input
X_train = np.array([x.reshape( (257, 1091, 1) ) for x in X_train])
X_test = np.array([x.reshape( (257, 1091, 1) ) for x in X_test])
X_eval = np.array([x.reshape( (257, 1091, 1) ) for x in X_eval])

# One-Hot encoding for classes
y_train_saved = y_train
#y_train = np.array(tf.keras.utils.to_categorical(y_train, 2))
y_test_saved = y_test
#y_test = np.array(tf.keras.utils.to_categorical(y_test, 2))
y_eval_saved = y_eval
#y_eval = np.array(tf.keras.utils.to_categorical(y_eval, 2))

[model, weight_model] = build_model()

print("model ready!!!!")

model.summary()

adam = optimizers.Adam(amsgrad=True)

model.compile(
        optimizer=adam,
        loss="binary_crossentropy",
        metrics=['accuracy'])

#model.compile(loss=[losses.binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

model.fit(
        x=X_train,
        y=y_train,
        epochs=30,
        batch_size=4,
        verbose=2,
        validation_data= (X_test, y_test), callbacks=[tensorboard, developmentSetEval(y_test_saved)])

#lr_reducer, early_stopper

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
    final.append(pred[0])

fpr, tpr, threshold = roc_curve(y_eval_saved, final, pos_label=1)
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)

print("Validation of best model!!!!")
del model
#model = load_model(DATADIR+"/weights/"+NAME+".hdf5", custom_objects={'tf': tf})
[model, weight_model] = build_model()
model.load_weights(DATADIR+"/weights/attention-1572888755.hdf5")

model.compile(
        optimizer=adam,
        loss="binary_crossentropy",
        metrics=['accuracy'])

score = model.evaluate( x=X_eval, y=y_eval, verbose=0)

print('Eval loss:', score[0])
print('Eval accuracy:', score[1])

predictions = model.predict(x=X_eval)

final = []
f = open(os.path.join(DATADIR, "predictions.txt"), "w+")
for pred in predictions:
    f.write(str(pred[0]) + "\n")
    final.append(pred[0])
f.close()

fpr, tpr, threshold = roc_curve(y_eval_saved, final, pos_label=1)
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)

EER = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1)
print(EER)

weight_model.save_weights(DATADIR+"/weights/weight_model_new_attention.hdf5")

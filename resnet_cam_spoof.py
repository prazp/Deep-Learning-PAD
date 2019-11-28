#!/usr/bin/python3

import keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, TimeDistributed, LSTM, SimpleRNN, Masking, GRU, BatchNormalization
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
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import roc_curve, auc
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
import resnet

# import librosa
# import librosa.display
# import cv2

#DATADIR = "/mnt/c/Users/prasa/code/Thesis"
DATADIR = "../../../g/data1a/wa66/Prasanth"

def train():

	NAME = "resnet-{}".format(int(time.time()))

	tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
	early_stopper = EarlyStopping(min_delta=0.001, patience=10)
	#lr_reducer = ReduceLROnPlateau(min_lr=0.5e-6)

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

	print("Data Ready for Model")

	model = resnet.ResnetBuilder.build_resnet_18((1, 1025, 63), 2)

	model.compile(
		optimizer="Adam",
		loss="categorical_crossentropy",
		metrics=['accuracy'])

	checkpoint_path=DATADIR+"/weights/weights_resnet_spoof.{epoch:02d}-{val_loss:.2f}.hdf5"
	checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')

	model.fit(
		x=X_train,
		y=y_train,
	    epochs=150,
	    batch_size=32,
	    validation_data= (X_test, y_test), callbacks=[early_stopper, tensorboard, checkpoint]) #lr_reducer

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

	# EER reference: https://yangcha.github.io/EER-ROC/
	fpr, tpr, threshold = roc_curve(y_test_saved, final, pos_label=1)
	EER = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1)

	# fnr = 1 - tpr
	# EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	print(EER)

def produce_weighted_grams(model_path):
	#model = load_model(model_path)
	
	sess = tf.Session()
	adam = optimizers.Adam(amsgrad=True)
	model = resnet.ResnetBuilder.build_resnet_18((1, 257, 1091), 2)
	model.load_weights(model_path)
	#model.summary()

	model.compile(
        optimizer=adam,
        loss="categorical_crossentropy",
        metrics=['accuracy'])

	# pickle_in = open(os.path.join(DATADIR, "specTrain_win400_hop160_cmvn.pickle"), "rb")
	# specTrain = pickle.load(pickle_in)
	# pickle_in.close()

	# X_train = []
	# y_train = []
	# for res in specTrain:
	# 	X_train.append(res[0])
	# 	y_train.append(res[1])

	# #Development Data
	# pickle_in = open(os.path.join(DATADIR, "specDev_win400_hop160_cmvn.pickle"), "rb")
	# specDev = pickle.load(pickle_in)
	# pickle_in.close()

	# X_test = []
	# y_test = []
	# for res in specDev:
	# 	X_test.append(res[0])
	# 	y_test.append(res[1])


	#Evaluation Data
	pickle_in = open(os.path.join(DATADIR, "specEval_win400_hop160_cmvn.pickle"), "rb")
	specEval = pickle.load(pickle_in)
	pickle_in.close()

	X_eval = []
	y_eval = []
	for res in specEval:
		X_eval.append(res[0])
		y_eval.append(res[1])

	#need to normalise the data
	# X_train_norm = tf.keras.utils.normalize(X_train, axis=1)
	# X_test_norm = tf.keras.utils.normalize(X_test, axis=1)

	# Reshape for CNN input
	#X_train = np.array([x.reshape( (257, 1091, 1) ) for x in X_train])
	#X_test = np.array([x.reshape( (257, 1091, 1) ) for x in X_test])
	X_eval = np.array([x.reshape( (257, 1091, 1) ) for x in X_eval])

	X_train_final = []
	j = 0

	class_weights = model.layers[-1].get_weights()[0]
	final_conv_layer = model.get_layer('activation_17')
	get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])

	# for utterance in X_train:
	# 	utterance = np.expand_dims(utterance, axis=0)
		
	# 	#Get the 512 input weights to the softmax.
	# 	[conv_outputs, predictions] = get_output([utterance])
	# 	conv_outputs = conv_outputs[0, :, :, :]

	# 	argmax = np.argmax(predictions) #index of the prediction
	# 	#Create the class activation map.
	# 	print(predictions)
	# 	cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
	# 	for i, w in enumerate(class_weights[:, argmax]):
	# 		cam += w * conv_outputs[:, :, i]
	# 	cam /= np.max(np.abs(cam))
		
	# 	#cam = cv2.resize(cam, (1091, 257))

	# 	cam = np.expand_dims(cam, axis=-1)
	# 	cam = tf.image.resize_images(cam, [257, 1091])
	# 	with sess.as_default():
	# 		cam = cam.eval()
	# 	cam = np.squeeze(cam)

	# 	X_train_final.append(np.multiply(np.squeeze(np.abs(X_train[j])),cam))
	# 	j += 1

	# pickle_out = open(os.path.join(DATADIR, "specTrain_win400_hop160_cmvn_weighted.pickle"), "wb")
	# pickle.dump(X_train_final, pickle_out)
	# pickle_out.close()

	# X_test_final = []
	# j = 0

	# for utterance in X_test:
	# 	utterance = np.expand_dims(utterance, axis=0)
		
	# 	#Get the 512 input weights to the softmax.
	# 	[conv_outputs, predictions] = get_output([utterance])
	# 	conv_outputs = conv_outputs[0, :, :, :]

	# 	argmax = np.argmax(predictions) #index of the prediction
	# 	#Create the class activation map.
	# 	print(predictions)
	# 	cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
	# 	for i, w in enumerate(class_weights[:, argmax]):
	# 		cam += w * conv_outputs[:, :, i]
	# 	cam /= np.max(np.abs(cam))

	# 	cam = np.expand_dims(cam, axis=-1)
	# 	cam = tf.image.resize_images(cam, [257, 1091])
	# 	with sess.as_default():
	# 		cam = cam.eval()
	# 	cam = np.squeeze(cam)
		
	# 	#cam = cv2.resize(cam, (1091, 257))

	# 	X_test_final.append(np.multiply(np.squeeze(np.abs(X_test[j])),cam))
	# 	j += 1

	# pickle_out = open(os.path.join(DATADIR, "specDev_win400_hop160_cmvn_weighted.pickle"), "wb")
	# pickle.dump(X_test_final, pickle_out)
	# pickle_out.close()

	X_eval_final = []
	j = 0

	for utterance in X_eval:
		utterance = np.expand_dims(utterance, axis=0)
		
		#Get the 512 input weights to the softmax.
		[conv_outputs, predictions] = get_output([utterance])
		conv_outputs = conv_outputs[0, :, :, :]

		argmax = np.argmax(predictions) #index of the prediction
		#Create the class activation map.
		print(predictions)
		cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
		for i, w in enumerate(class_weights[:, argmax]):
			cam += w * conv_outputs[:, :, i]
		cam /= np.max(np.abs(cam))

		cam = np.expand_dims(cam, axis=-1)
		cam = tf.image.resize_images(cam, [257, 1091])
		with sess.as_default():
			cam = cam.eval()
		cam = np.squeeze(cam)

		#cam = cv2.resize(cam, (1091, 257))

		X_eval_final.append(np.multiply(np.squeeze(np.abs(X_eval[j])),cam))
		j += 1

	pickle_out = open(os.path.join(DATADIR, "specEval_win400_hop160_cmvn_weighted.pickle"), "wb")
	pickle.dump(X_eval_final, pickle_out)
	pickle_out.close()

def visualize_class_activation_map(model_path, output_path):
	#model = load_model(model_path)

	sess = tf.Session()
	adam = optimizers.Adam(amsgrad=True)
	model = resnet.ResnetBuilder.build_resnet_18((1, 257, 1091), 2)
	model.load_weights(model_path)
	#model.summary()

	model.compile(
        optimizer=adam,
        loss="categorical_crossentropy",
        metrics=['accuracy'])

	pickle_in = open(os.path.join(DATADIR, "specTrain_win400_hop160_cmvn.pickle"), "rb")
	specTrain = pickle.load(pickle_in)
	pickle_in.close()

	X_train = []
	y_train = []
	for res in specTrain:
		X_train.append(res[0])
		y_train.append(res[1])

	original_img = X_train[0]
	img = np.array(original_img.reshape( (257, 1091, 1) ))
	img = np.expand_dims(img, axis=0)
	
	#Get the 512 input weights to the softmax.
	class_weights = model.layers[-1].get_weights()[0]
	final_conv_layer = model.get_layer('activation_16')
	get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
	[conv_outputs, predictions] = get_output([img])
	print(predictions)
	conv_outputs = conv_outputs[0, :, :, :]

	argmax = np.argmax(predictions) #index of the prediction
	#Create the class activation map.
	cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
	for i, w in enumerate(class_weights[:, argmax]):
		cam += w * conv_outputs[:, :, i]
	cam /= np.max(np.abs(cam))
	
	cam = np.expand_dims(cam, axis=-1)
	cam = tf.image.resize_images(cam, [257, 1091])
	with sess.as_default():
		cam = cam.eval()
	cam = np.squeeze(cam)
	
	#cam = cv2.resize(cam, (1091, 257))
	
	print(cam)
	heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
	#heatmap[np.where(cam < 0.2)] = 0
	img = np.uint8(255*cam)
	print("Writing to output")

	cv2.imwrite(output_path, img)

	plt.pcolormesh(original_img)
	plt.title('Power spectrogram genuine utterance')
	plt.colorbar(orientation='vertical')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.tight_layout()
	plt.savefig('original.png')
	plt.clf()

	plt.pcolormesh(np.multiply(np.abs(original_img),cam),)
	plt.title('Power spectrogram genuine utterance')
	plt.colorbar(orientation='vertical')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.tight_layout()
	plt.savefig('new_img.png')
	plt.clf()

if __name__ == '__main__':
    #train()
    #visualize_class_activation_map(DATADIR+"/resnet_for_weighting-1571131680.hdf5", DATADIR+"/heatmap.bmp")
    produce_weighted_grams(DATADIR+"/resnet_for_weighting-1571131680.hdf5")

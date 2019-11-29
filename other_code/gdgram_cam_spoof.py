#!/usr/bin/python3

import keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, TimeDistributed, LSTM, SimpleRNN, Masking, GRU, average, BatchNormalization
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
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_curve, auc
from tensorflow.python.keras import backend as K
import h5py
import resnet
# import librosa
# import librosa.display
import cv2

DATADIR = "/mnt/c/Users/prasa/code/Thesis"
#DATADIR = "../../../g/data1a/wa66/Prasanth"

def train():
	NAME = "group_delay_dnn_basic_speech-{}".format(int(time.time()))

	early_stopper = EarlyStopping(min_delta=0.001, patience=10)
	lr_reducer = ReduceLROnPlateau()
	tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

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

def produce_weighted_grams(model_path):
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
		#devSet = np.transpose(np.array(matFile['devSet']))

	print("Extracted data from mat file")

	training = []
	i = 0
	for x in trainSet:
		training.append((x, trn_label[i]))
		i += 1

	X_train = []
	y_train = []

	for res in training:
		X_train.append(res[0])
		y_train.append(res[1])

	# X_test = []
	# y_test = []
	# for x in devSet:
	# 	X_test.append(x)

	# for y in dev_label:
	# 	y_test.append(y)

	print("Data Ready for Model")

	model = load_model(model_path)
	#model.summary()

	print("Model loaded")

	#need to normalise the data
	X_train = np.asarray(X_train)
	print(X_train.shape)
	X_train_norm = tf.keras.utils.normalize(X_train, axis=1)
	#X_test_norm = tf.keras.utils.normalize(X_test, axis=1)

	print("Normalised data")

	# Reshape for CNN input
	X_train_norm = np.array([x.reshape( (513, 256, 1) ) for x in X_train_norm])
	#X_test_norm = np.array([x.reshape( (513, 256, 1) ) for x in X_test_norm])

	X_train_final = []
	j = 0

	class_weights = model.layers[-1].get_weights()[0]
	final_conv_layer = model.get_layer('activation_17')
	get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])

	for utterance in X_train_norm:
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
		#cam = tf.image.resize_images(cam, [513, 256])
		cam = cv2.resize(cam, (256, 513))

		X_train_final.append(np.multiply(np.abs(X_train[j]),cam))
		j += 1

	pickle_out = open(os.path.join(DATADIR, "gdgram_weighted_Train.pickle"), "wb")
	pickle.dump(X_train_final, pickle_out)
	pickle_out.close()

	# X_test_final = []
	# j = 0

	# for utterance in X_test_norm:
	# 	utterance = np.expand_dims(utterance, axis=0)
		
	# 	#Get the 512 input weights to the softmax.
	# 	class_weights = model.layers[-1].get_weights()[0]
	# 	final_conv_layer = model.get_layer('activation_17')
	# 	get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
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
	# 	cam = tf.image.resize_images(cam, [513, 256])
	# 	#cam = cv2.resize(cam, (256, 513))

	# 	X_test_final.append(np.multiply(np.abs(X_test[j]),cam))
	# 	j += 1

	# pickle_out = open(os.path.join(DATADIR, "gdgram_weighted_Test.pickle"), "wb")
	# pickle.dump(X_test_final, pickle_out)
	# pickle_out.close()

def visualize_class_activation_map(model_path, output_path):
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

	# #Development Data
	# pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_dev.trl.pickle"), "rb")
	# dev_label_array = pickle.load(pickle_in)
	# pickle_in.close()

	# #spoof is defined as 0 and genuine defined as 1
	# dev_label = []
	# for i in dev_label_array:
	# 	if i[1] == b'spoof':
	# 		dev_label.append(0)
	# 	else:
	# 		dev_label.append(1)

	with h5py.File(DATADIR+"/modified_group_delay.mat", 'r') as matFile:
		trainSet = np.transpose(np.array(matFile['trainSet']))
		#devSet = np.transpose(np.array(matFile['devSet']))

	print("Extracted data from mat file")

	training = []
	i = 0
	for x in trainSet:
		training.append((x, trn_label[i]))
		i += 1

	X_train = []
	y_train = []

	for res in training:
		X_train.append(res[0])
		y_train.append(res[1])

	# X_test = []
	# y_test = []
	# for x in devSet:
	# 	X_test.append(x)

	# for y in dev_label:
	# 	y_test.append(y)

	print("Data Ready for Model")

	#need to normalise the data
	original_img = X_train[0]

	X_train = tf.keras.utils.normalize(X_train, axis=1)
	#X_test = tf.keras.utils.normalize(X_test, axis=1)

	model = load_model(model_path)
	#model.summary()

	#Visualize class activation map

	img = X_train[0]
	img = np.array(img.reshape( (513, 256, 1) ))
	img = np.expand_dims(img, axis=0)
	
	#Get the 512 input weights to the softmax.
	class_weights = model.layers[-1].get_weights()[0]
	final_conv_layer = model.get_layer('activation_17')
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
	cam = cv2.resize(cam, (256, 513))
	print(cam)
	heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
	#heatmap[np.where(cam < 0.2)] = 0
	img = heatmap
	print("Writing to output")

	cv2.imwrite(output_path, img)

	librosa.display.specshow(np.abs(original_img), y_axis='log', x_axis='time', sr=16000)
	plt.title('Power spectrogram original')
	plt.colorbar()
	plt.tight_layout()
	plt.savefig(DATADIR+"/original_img.png")
	plt.clf()

	librosa.display.specshow(np.multiply(np.abs(original_img),cam), y_axis='log', x_axis='time', sr=16000)
	plt.title('Power spectrogram new')
	plt.colorbar()
	plt.tight_layout()
	plt.savefig(DATADIR+"/new_img.png")

def test(model_path):
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
		devSet = np.transpose(np.array(matFile['devSet']))

	print("Extracted data from mat file")

	X_test = []
	y_test = []
	for x in devSet:
		X_test.append(x)

	for y in dev_label:
		y_test.append(y)

	print("Data Ready for Model")

	#need to normalise the data
	X_test = tf.keras.utils.normalize(X_test, axis=1)

	X_test = np.array([x.reshape( (513, 256, 1) ) for x in X_test])

	# One-Hot encoding for classes
	y_test_saved = y_test
	y_test = np.array(keras.utils.to_categorical(y_test, 2))

	model = load_model(model_path)

	predictions = model.predict(x=X_test)

	final = []
	f = open(os.path.join(DATADIR, "predictions_gdgram_cam.txt"), "w+")
	for pred in predictions:
		f.write(str(pred[0]) + " " + str(pred[1]) + "\n")
		if pred[1] != 0:
			final.append(math.log(pred[1], 10))
		else:
			final.append(pred[1])
	f.close()

	# EER reference: https://yangcha.github.io/EER-ROC/
	fpr, tpr, threshold = roc_curve(y_test_saved, final, pos_label=1)
	EER = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1)

	fnr = 1 - tpr
	OLD_EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	print(OLD_EER)
	print(EER)

if __name__ == '__main__':
    #train()
    #test(DATADIR+"/weights_gdgram_spoof.15-4.42.hdf5")
    #visualize_class_activation_map(DATADIR+"/weights_gdgram_spoof.15-4.42.hdf5", DATADIR+"/heatmap.bmp")
    produce_weighted_grams(DATADIR+"/weights_gdgram_spoof.15-4.42.hdf5")
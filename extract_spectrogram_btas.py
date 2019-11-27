#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import pickle
import random
import time
import math
from scipy.signal import stft
import speechpy
from sklearn import preprocessing

DATADIR = "/mnt/c/Users/prasa/code/Thesis/data_btas"

length = 16000*5

print("Starting Extracting Data - Training...")

fd = open(os.path.join(DATADIR, "train_genuine.txt"), 'r')
trn_label_array_genuine = np.loadtxt(fd, dtype={'names': ('col1', 'col2', 'col3'),
'formats': ('S3', 'S60', 'S7')})

#spoof is defined as 0 and genuine defined as 1
specTrain = []
smallest = 16000*30
largest = 0
num_over = 0
for i in trn_label_array_genuine:
	fs, data = wavfile.read(os.path.join(DATADIR,'train',i[1].decode('utf-8')))

	if (len(data) < smallest):
		smallest = len(data)

	if (len(data) > largest):
		largest = len(data)

	if (len(data) > length):
		num_over += 1

	if (len(data) < length):
		data = np.pad(data,(0,length-len(data)),'wrap')
	
	ps = stft(data[0:length], nperseg=400, noverlap=240, nfft=512, boundary='even', fs=16000) #hop_length = 160, win_length = 400
	result = np.log10(np.abs(ps[2])**2)
	result[result==-np.inf] = 0
	result = preprocessing.StandardScaler().fit_transform(np.transpose(result))
	specTrain.append((np.transpose(result), 1))

fd = open(os.path.join(DATADIR, "train_spoof.txt"), 'r')
trn_label_array_spoof = np.loadtxt(fd, dtype={'names': ('col1', 'col2', 'col3'),
'formats': ('S3', 'S110', 'S7')})
index = 0
for i in trn_label_array_spoof:
	fs, data = wavfile.read(os.path.join(DATADIR,'train',i[1].decode('utf-8')))

	if (len(data) < smallest):
		smallest = len(data)

	if (len(data) > largest):
		largest = len(data)

	if (len(data) > length):
		num_over += 1

	if (len(data) < length):
		data = np.pad(data,(0,length-len(data)),'wrap')
	
	ps = stft(data[0:length], nperseg=400, noverlap=240, nfft=512, boundary='even', fs=16000) #hop_length = 160, win_length = 400
	result = np.log10(np.abs(ps[2])**2)
	result[result==-np.inf] = 0
	result = preprocessing.StandardScaler().fit_transform(np.transpose(result))
	specTrain.append((np.transpose(result), 0))
	index += 1
	if index > 2800:
		break

print("Training Data Finished - Saving...")

pickle_out = open(os.path.join(DATADIR, "BTAS2016_Spectrain.pickle"), "wb")
pickle.dump(specTrain, pickle_out)
pickle_out.close()

print(smallest)
print(largest)
print(num_over)

print("Starting Extracting Data - Development...")

fd = open(os.path.join(DATADIR, "dev_genuine.txt"), 'r')
dev_label_array_genuine = np.loadtxt(fd, dtype={'names': ('col1', 'col2', 'col3'),
'formats': ('S3', 'S60', 'S7')})

#spoof is defined as 0 and genuine defined as 1
specDev = []
smallest = 16000*30
largest = 0
num_over = 0
for i in dev_label_array_genuine:
	fs, data = wavfile.read(os.path.join(DATADIR,'dev',i[1].decode('utf-8')))

	if (len(data) < smallest):
		smallest = len(data)

	if (len(data) > largest):
		largest = len(data)

	if (len(data) > length):
		num_over += 1

	if (len(data) < length):
		data = np.pad(data,(0,length-len(data)),'wrap')
	
	ps = stft(data[0:length], nperseg=400, noverlap=240, nfft=512, boundary='even', fs=16000) #hop_length = 160, win_length = 400
	result = np.log10(np.abs(ps[2])**2)
	result[result==-np.inf] = 0
	result = preprocessing.StandardScaler().fit_transform(np.transpose(result))
	specDev.append((np.transpose(result), 1))

fd = open(os.path.join(DATADIR, "dev_spoof.txt"), 'r')
dev_label_array_spoof = np.loadtxt(fd, dtype={'names': ('col1', 'col2', 'col3'),
'formats': ('S3', 'S110', 'S7')})

index = 0
for i in dev_label_array_spoof:
	fs, data = wavfile.read(os.path.join(DATADIR,'dev',i[1].decode('utf-8')))

	if (len(data) < smallest):
		smallest = len(data)

	if (len(data) > largest):
		largest = len(data)

	if (len(data) > length):
		num_over += 1

	if (len(data) < length):
		data = np.pad(data,(0,length-len(data)),'wrap')
	
	ps = stft(data[0:length], nperseg=400, noverlap=240, nfft=512, boundary='even', fs=16000) #hop_length = 160, win_length = 400
	result = np.log10(np.abs(ps[2])**2)
	result[result==-np.inf] = 0
	result = preprocessing.StandardScaler().fit_transform(np.transpose(result))
	specDev.append((np.transpose(result), 0))
	index += 1
	if index > 2800:
		break

print("Development Data Finished - Saving...")

pickle_out = open(os.path.join(DATADIR, "BTAS2016_Specdev.pickle"), "wb")
pickle.dump(specDev, pickle_out)
pickle_out.close()

print(smallest)
print(largest)
print(num_over)

print("Starting Extracting Data - Evaluation...")

fd = open(os.path.join(DATADIR, "test" ,"test_genuine_list_derandomized.txt"), 'r')
eval_label_array_genuine = np.loadtxt(fd, dtype={'names': ('col1', 'col2', 'col3'),
'formats': ('S3', 'S60', 'S7')})

#spoof is defined as 0 and genuine defined as 1
specEval = []
smallest = 16000*30
largest = 0
num_over = 0
index = 0
for i in eval_label_array_genuine:
	fs, data = wavfile.read(os.path.join(DATADIR,'test/test_corrected',i[1].decode('utf-8')))

	if (len(data) < smallest):
		smallest = len(data)

	if (len(data) > largest):
		largest = len(data)

	if (len(data) > length):
		num_over += 1

	if (len(data) < length):
		data = np.pad(data,(0,length-len(data)),'wrap')
	
	ps = stft(data[0:length], nperseg=400, noverlap=240, nfft=512, boundary='even', fs=16000) #hop_length = 160, win_length = 400
	result = np.log10(np.abs(ps[2])**2)
	result[result==-np.inf] = 0
	result = preprocessing.StandardScaler().fit_transform(np.transpose(result))
	specEval.append((np.transpose(result), 1))
	index += 1
	if index >= 6309:
		break

fd = open(os.path.join(DATADIR, "test", "test_spoof_list_derandomized.txt"), 'r')
eval_label_array_spoof = np.loadtxt(fd, dtype={'names': ('col1', 'col2', 'col3'),
'formats': ('S3', 'S110', 'S6')})

count = 0
index = 0
for i in eval_label_array_spoof:
	index += 1
	if index > 49660:
		break
	if i[2].decode('utf-8') != 'replay':
		continue

	fs, data = wavfile.read(os.path.join(DATADIR,'test/test_corrected',i[1].decode('utf-8')))

	if (len(data) < smallest):
		smallest = len(data)

	if (len(data) > largest):
		largest = len(data)

	if (len(data) > length):
		num_over += 1

	if (len(data) < length):
		data = np.pad(data,(0,length-len(data)),'wrap')
	
	ps = stft(data[0:length], nperseg=400, noverlap=240, nfft=512, boundary='even', fs=16000) #hop_length = 160, win_length = 400
	result = np.log10(np.abs(ps[2])**2)
	result[result==-np.inf] = 0
	result = preprocessing.StandardScaler().fit_transform(np.transpose(result))
	specEval.append((np.transpose(result), 0))
	count += 1

print("Development Data Finished - Saving...")

pickle_out = open(os.path.join(DATADIR, "BTAS2016_Speceval.pickle"), "wb")
pickle.dump(specEval, pickle_out)
pickle_out.close()

print(count)
print(smallest)
print(largest)
print(num_over)

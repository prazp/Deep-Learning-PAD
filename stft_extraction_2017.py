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

#DATADIR = "/mnt/c/Users/prasa/code/Thesis"
DATADIR = "../../../g/data1a/wa66/Prasanth"

length = 16000*5

# Training Data
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
specTrain = []

print("Extracting Training Data")

for i in range(len(trn_data)):
	if (len(trn_data[i]) < length):
		trn_data[i] = np.pad(trn_data[i],(0,length-len(trn_data[i])),'wrap')
	
	ps = stft(trn_data[i][0:length], nperseg=400, noverlap=240, nfft=512, boundary='even', fs=16000) #hop_length = 160, win_length = 400
	result = np.log10(np.abs(ps[2])**2)
	result[result==-np.inf] = 0
	result = preprocessing.StandardScaler().fit_transform(np.transpose(result))
	#result = speechpy.processing.cmvnw(np.transpose(result),win_size=301,variance_normalization=True)
	specTrain.append((np.transpose(result), trn_label[i]))

print("Saving Data!!!")

pickle_out = open(os.path.join(DATADIR, "specTrain_win400_hop160_5s.pickle"), "wb")
pickle.dump(specTrain, pickle_out)
pickle_out.close()

# plt.pcolormesh(ps[1], ps[0], specTrain[0][0])
# plt.title('Power spectrogram genuine utterance')
# plt.colorbar(orientation='vertical')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.tight_layout()
# plt.savefig('genuine_scipy_standard_scalar.png')
# plt.clf()

# plt.pcolormesh(ps[1], ps[0], specTrain[0][0])
# plt.title('Power spectrogram spoofed utterance')
# plt.colorbar(orientation='vertical')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.tight_layout()
# plt.savefig('spoofed_stft_standard_scalar_old.png')
# plt.clf()

print("Extracting Dev Data")

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
specDev = []

for i in range(len(dev_data)):
	if (len(dev_data[i]) < length):
		dev_data[i] = np.pad(dev_data[i],(0,length-len(dev_data[i])),'wrap')
	ps = stft(dev_data[i][0:length], nperseg=400, noverlap=240, nfft=512, boundary='even', fs=16000) #hop_length = 160, win_length = 400
	result = np.log10(np.abs(ps[2])**2)
	result[result==-np.inf] = 0
	result = preprocessing.StandardScaler().fit_transform(np.transpose(result))
	#result = speechpy.processing.cmvnw(np.transpose(result),win_size=301,variance_normalization=True)
	specDev.append((np.transpose(result), dev_label[i]))

print("Saving Data")

pickle_out = open(os.path.join(DATADIR, "specDev_win400_hop160_5s.pickle"), "wb")
pickle.dump(specDev, pickle_out)
pickle_out.close()

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
specEval = []

print("Extracting Training Data")

for i in range(len(eval_data)):
	if (len(eval_data[i]) < length):
		eval_data[i] = np.pad(eval_data[i],(0,length-len(eval_data[i])),'wrap')
	ps = stft(eval_data[i][0:length], nperseg=400, noverlap=240, nfft=512, boundary='even', fs=16000) #hop_length = 160, win_length = 400
	result = np.log10(np.abs(ps[2])**2)
	result[result==-np.inf] = 0
	result = preprocessing.StandardScaler().fit_transform(np.transpose(result))
	#result = speechpy.processing.cmvnw(np.transpose(result),win_size=301,variance_normalization=True)
	specEval.append((np.transpose(result), eval_label[i]))

print("Saving Data!!!")

pickle_out = open(os.path.join(DATADIR, "specEval_win400_hop160_5s.pickle"), "wb")
pickle.dump(specEval, pickle_out)
pickle_out.close()

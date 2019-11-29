#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import pickle
import random
from python_speech_features import fbank, logfbank
import time
import librosa.display
import math

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
fbankTrain = []

for i in [0]:
	#if (len(trn_data[i]) < 16000):
	#	trn_data[i] = np.pad(trn_data[i],(0,16000-len(trn_data[i])),'wrap')
	ps = logfbank(trn_data[i].astype(float), nfilt=120)
	fbankTrain.append((ps, trn_label[i]))

plt.pcolor(np.linspace(0,len(trn_data[0])/16000,241), range(120), np.transpose(fbankTrain[0][0]))
plt.title('Mel Filter Bank Genuine Utterance')
plt.colorbar(orientation='vertical')
plt.ylabel('Filter Num')
plt.xlabel('Time [sec]')
plt.savefig('genuine_mel_fbank.png')
plt.clf()

# librosa.display.specshow(librosa.amplitude_to_db(np.abs(fbankTrain[0][0]),ref=np.max), y_axis='log', x_axis='time')
# plt.title('Power spectrogram genuine')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.savefig('genuine.png')
# plt.clear()

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

# pickle_in = open(os.path.join(DATADIR, "ASVspoof2017_V2_dev.pickle"), "rb")
# dev_data = pickle.load(pickle_in)
# pickle_in.close()
# fbankDev = []

# for i in range(len(dev_data)):	
# 	if (len(dev_data[i]) < 16000):
# 		dev_data[i] = np.pad(dev_data[i],(0,16000-len(dev_data[i])),'wrap')
# 	ps = fbank(dev_data[i].astype(float)[0:16000], nfilt=52)
# 	fbankDev.append((ps, dev_label[i]))

# # need to do the following as I shuffle the data
# # random.shuffle(specTrain)

# print("Saving Data")

# pickle_out = open(os.path.join(DATADIR, "fbankTrain.pickle"), "wb")
# pickle.dump(fbankTrain, pickle_out)
# pickle_out.close()

# pickle_out = open(os.path.join(DATADIR, "fbankDev.pickle"), "wb")
# pickle.dump(fbankDev, pickle_out)
# pickle_out.close()

# librosa.display.specshow(librosa.amplitude_to_db(np.abs(specTrain[1508][0]),ref=np.max), y_axis='log', x_axis='time')
# plt.title('Power spectrogram spoofed')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.savefig('spoof.png')
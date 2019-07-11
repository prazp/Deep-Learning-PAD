#!/usr/bin/python3

import numpy as np
import os
import pickle
import random
import librosa
import time
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
melResultsTrain = []

for i in range(len(trn_data)):
	j = 0
	ps = []
	while j < len(trn_data[i]):
		mfcc = librosa.feature.mfcc(trn_data[i][j:j+320].astype(float), n_mfcc=20, sr=16000)
		ps.append(mfcc);
		j += 320
	melResultsTrain.append((ps, trn_label[i]))

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
	while j < len(trn_data[i]):
		mfcc = librosa.feature.mfcc(trn_data[i][j:j+320].astype(float), n_mfcc=20, sr=16000)
		ps.append(mfcc);
		j += 320
	melResultsDev.append((ps, dev_label[i]))

#need to do the following as I shuffle the data
random.shuffle(melResultsTrain)

pickle_out = open(os.path.join(DATADIR, "melResultsTrain.pickle"), "wb")
pickle.dump(melResultsTrain, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATADIR, "melResultsDev.pickle"), "wb")
pickle.dump(melResultsDev, pickle_out)
pickle_out.close()

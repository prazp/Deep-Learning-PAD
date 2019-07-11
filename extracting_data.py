#!/usr/bin/python3

from scipy.io import wavfile
import os
import numpy as np
import pickle

DATADIR = "/mnt/c/Users/prasa/code/Thesis/ASVspoof2017"

DATACATEGORIES = ["ASVspoof2017_V2_train", "ASVspoof2017_V2_dev", "ASVspoof2017_V2_eval"]

LABELCATEGORIES = ["ASVspoof2017_V2_train.trn", "ASVspoof2017_V2_dev.trl", "ASVspoof2017_V2_eval.trl"]

data_extraction = 1
label_extraction = 0

train = []
devel = []
evalu = []

if data_extraction:
	for category in DATACATEGORIES:

		#which data is required to be extracted
		path = os.path.join(DATADIR, category)
		for wav in os.listdir(path):
			fs, data = wavfile.read(os.path.join(path,wav))
			if category == DATACATEGORIES[0]:
				train.append(data)
			if category == DATACATEGORIES[1]:
				devel.append(data)
			if len(DATACATEGORIES) == 3:
				if category == DATACATEGORIES[2]:
					evalu.append(data)
		print("Finished " + category)

	#extracting and serialising relavent data
	if (train):
		pickle_out = open(DATACATEGORIES[0]+".pickle", "wb")
		pickle.dump(train, pickle_out)
		pickle_out.close()

	if (devel):
		pickle_out = open(DATACATEGORIES[1]+".pickle", "wb")
		pickle.dump(devel, pickle_out)
		pickle_out.close()

	if (evalu):
		pickle_out = open(DATACATEGORIES[2]+".pickle", "wb")
		pickle.dump(evalu, pickle_out)
		pickle_out.close()

#extracting labels from metadata
if label_extraction:
	for file in os.listdir(os.path.join(DATADIR, "protocol_V2")):
		if file == LABELCATEGORIES[0]+".txt":
			fd = open(os.path.join(DATADIR, "protocol_V2", file), 'r')
			pickle_out = open(LABELCATEGORIES[0]+".pickle", "wb")
			pickle.dump(np.loadtxt(fd, dtype={'names': ('col1', 'col2', 'col3', 'col4', 'col5', 'col6'),
           'formats': ('S13', 'S8', 'S8', 'S8', 'S8', 'S8')}), pickle_out)
			pickle_out.close()			
		if file == LABELCATEGORIES[1]+".txt":
			fd = open(os.path.join(DATADIR, "protocol_V2", file), 'r')
			pickle_out = open(LABELCATEGORIES[1]+".pickle", "wb")
			pickle.dump(np.loadtxt(fd, dtype={'names': ('col1', 'col2', 'col3', 'col4', 'col5', 'col6'),
           'formats': ('S13', 'S8', 'S8', 'S8', 'S8', 'S8')}), pickle_out)
			pickle_out.close()

		if len(LABELCATEGORIES) == 3:
			if file == LABELCATEGORIES[2]+".txt":
				fd = open(os.path.join(DATADIR, "protocol_V2", file), 'r')
				pickle_out = open(LABELCATEGORIES[2]+".pickle", "wb")
				pickle.dump(np.loadtxt(fd, dtype={'names': ('col1', 'col2', 'col3', 'col4', 'col5', 'col6'),
	           'formats': ('S13', 'S8', 'S8', 'S8', 'S8', 'S8')}), pickle_out)
				pickle_out.close()

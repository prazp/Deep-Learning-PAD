#!/usr/bin/python3

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import resource, sys

fs = 16000

sin_files_genuine = []

DATADIR = "/mnt/c/Users/prasa/code/Thesis"

print("genuine creation")

gain_bin = np.linspace(1, 5, 9)
offset_bin = [-1, 0, 1]
phase_bin = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]

#Generate genine sine waves
for gain in range(len(gain_bin)):
	for phase in range(len(phase_bin)):
		for offset in range(len(offset_bin)):
			y = []
			for x in range(32000):
				y.append(gain_bin[gain]*np.sin(2 * np.pi * x / fs - 2 * math.pi * phase_bin[phase]/fs)+offset_bin[offset])
				#y.append(gain*math.sin(2 * math.pi * x/fs - 2 * math.pi * phase/fs))
			sin_files_genuine.append(y)


mean = 0
std =0.8
num_samples = 32000
sin_files_spoof = []

print("spoof creation")

#Generate spoofed sine waves
for wave in sin_files_genuine:
	samples = np.random.normal(mean, std, size=num_samples)
	temp = [wave[i]+samples[i] for i in range(len(wave))]
	sin_files_spoof.append(temp)

print("done!")

#Save Genuine and Spoofed sine waves
pickle_out = open(os.path.join(DATADIR, "genuine_sines.pickle"), "wb")
pickle.dump(sin_files_genuine, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATADIR, "spoof_sines.pickle"), "wb")
pickle.dump(sin_files_spoof, pickle_out)
pickle_out.close()

#Plotting Sine waves for debugging
x = list(range(0, 32000))

plt.plot(x, sin_files_genuine[0])
plt.ylabel('Amplitude')
plt.xlabel('sample(n)')
plt.savefig('genuine.png')

plt.plot(x, sin_files_spoof[0])
plt.ylabel('Amplitude')
plt.xlabel('sample(n)')
plt.savefig('spoof.png')
import numpy, scipy.io, os, pickle

DATADIR = "/mnt/c/Users/prasa/code/Thesis"

print("Training data conversion")

#training
pickle_in = open(os.path.join(DATADIR, "intermediate_output_train.pickle"), "rb")
train_data = pickle.load(pickle_in)
pickle_in.close()

train_genuine = []
train_spoof = []
i = 0
f = open(os.path.join(DATADIR, "ground_truth_train.txt"), "r") 
for x in f:
  if x == "1\n":
  	train_genuine.append(train_data[i])
  else:
  	train_spoof.append(train_data[i])

  i += 1

train_genuine_final = []
for i in train_genuine:
	for j in i:
		train_genuine_final.append(j)

train_spoof_final = []
for i in train_spoof:
	for j in i:
		train_spoof_final.append(j)

scipy.io.savemat(os.path.join(DATADIR, "train.mat"), mdict={'train_genuine': train_genuine_final, 'train_spoof': train_spoof_final})

print("Test data conversion")

#test
pickle_in = open(os.path.join(DATADIR, "intermediate_output_test.pickle"), "rb")
test_data = pickle.load(pickle_in)
pickle_in.close()

scipy.io.savemat(os.path.join(DATADIR, "test.mat"), mdict={'test_data': test_data})
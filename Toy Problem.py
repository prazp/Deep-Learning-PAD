#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import TensorBoard

#tensor board for visualisation
NAME = "mnistDatabase_speech-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

mnist = tf.keras.datasets.mnist #28x28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#scale between 0 and 1 
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#neural network architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


#optimizer we can use stochasitic gradient descent
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#we now train the model, each epoch is just the full pass through the entire dataset
model.fit(x_train,y_train, epochs=3, callbacks=[tensorboard])

#evaluating trained neural network
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
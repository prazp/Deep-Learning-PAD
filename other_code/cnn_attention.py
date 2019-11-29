#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

mnist = tf.keras.datasets.mnist #28x28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#scale between 0 and 1 
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


#optimizer we can use stochasitic gradient descent
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#epoch is just the full past through the entire dataset
model.fit(x_train,y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('my_first_model')

#new_model = tf.keras.models.load_model('my_first_model')
#predictions = new_model.predict([x_test])
#print(predictions)
#print(np.argmax(predictions[0]))

plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.savefig('image.png')

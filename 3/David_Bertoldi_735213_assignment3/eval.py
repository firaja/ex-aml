#!/usr/bin/env python

"""eval.py: Solution to the third assignment."""
__author__      = "David Bertoldi"


import tensorflow as tf
import torch 
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os

from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, Input, GaussianNoise, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.initializers import HeNormal, HeUniform
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)



def build_model(input_shape, classes, activation, initializer, regularizer, dropout):
	model = Sequential()
	model.add(Conv2D(8, (3, 3), activation=activation, kernel_initializer=initializer, padding='same', input_shape=input_shape))
	model.add(Conv2D(8, (3, 3), activation=activation, kernel_initializer=initializer, padding='same'))
	model.add(AveragePooling2D((2, 2)))
	model.add(Conv2D(16, (3, 3), activation=activation, kernel_initializer=initializer, padding='same'))
	model.add(AveragePooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dropout(dropout))
	model.add(Dense(classes, activation='softmax'))
	return model


def start():
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


	# Separate the test data
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=SEED, shuffle=True, stratify = None)

	#scaling
	X_train = X_train.astype('float32') / 255.
	X_val = X_val.astype('float32') / 255.
	X_test = X_test.astype('float32') / 255.

	X_train = np.expand_dims(X_train, -1)
	X_val = np.expand_dims(X_val, -1)
	X_test = np.expand_dims(X_test, -1)

	#one-hot encoding
	Y_train = np_utils.to_categorical(y_train)
	Y_val = np_utils.to_categorical(y_val)
	Y_test = np_utils.to_categorical(y_test)
	


	input_dims = X_test.shape[1:]
	nb_classes = Y_train.shape[1]


	print(input_dims)


	# configurations
	activation = LeakyReLU(alpha=0.01)
	optimizer = Adam(learning_rate=1e-03)
	regularizer = None
	initializer = HeNormal(seed=SEED)
	loss = 'categorical_crossentropy'
	dropout = 0.4
	e = 50
	bs = 256


	
	# model definition
	model = build_model(input_dims, nb_classes, activation, initializer, regularizer, dropout)
	model.load_weights('acc2.hdf5')
	model.compile(optimizer=optimizer, loss=loss, metrics=['categorical_accuracy'])

	
	score, acc = model.evaluate(X_test, Y_test, batch_size=bs)
	print('Test score:', score)
	print('Test accuracy:', acc)
	
	# confusion matrix
	Y_test_pred = model.predict(X_test)
	y_test_pred = Y_test_pred.argmax(1)
	cm = confusion_matrix(y_test, y_test_pred)
	fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(20,20), colorbar=True, cmap='GnBu')
	plt.show()


def format(s, activation, regularizer, loss, optimizer, e, bs, dropout):
	return s.format(activation if isinstance(activation, str) else type(activation).__name__, 
					type(regularizer).__name__, 
					loss, 
					type(optimizer).__name__, 
					e, 
					bs, 
					dropout)


if __name__ == '__main__':
	start()
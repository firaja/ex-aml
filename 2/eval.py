#!/usr/bin/env python

"""assignment2.py: Solution to the second assignment (no regularization)."""
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
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, Input, GaussianNoise, Dropout
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



BASE_URL = 'https://github.com/DBertazioli/multi-mnist_custom/raw/master/final_dataset/'
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)


def load_data():
	X_train_path = tf.keras.utils.get_file('X_train.npz', BASE_URL + 'X_train.npz')
	y_train_path = tf.keras.utils.get_file('y_train.npz', BASE_URL + 'y_train.npz')
	X_test_path = tf.keras.utils.get_file('X_test.npz', BASE_URL + 'X_test.npz')
	y_test_path = tf.keras.utils.get_file('y_test.npz', BASE_URL + 'y_test.npz')
	with np.load(X_train_path) as X_train, np.load(y_train_path) as y_train, np.load(X_test_path) as X_test, np.load(y_test_path) as y_test:
		return X_train["arr_0"], y_train["arr_0"], X_test["arr_0"], y_test["arr_0"]


def build_model(input_shape, classes, activation, initializer, regularizer, dropout):
	model = Sequential()
	model.add(Dense(80, activation = activation, kernel_initializer=initializer, kernel_regularizer=regularizer, input_shape=input_shape))
	model.add(Dropout(dropout))
	model.add(Dense(70, activation = activation,  kernel_regularizer=regularizer, kernel_initializer=initializer))
	model.add(Dropout(dropout))
	model.add(Dense(classes, activation = "softmax", kernel_regularizer=regularizer, kernel_initializer=initializer))
	return model


def start():
	X_train, y_train, X_test, y_test = load_data()

	print(y_train)

	num_of_lables = len(set(np.concatenate((y_train, y_test))))

	# Separate the test data
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED, shuffle=True, stratify = None)

	#scaling
	X_train = X_train.astype('float32') / 255.
	X_val = X_val.astype('float32') / 255.
	X_test = X_test.astype('float32') / 255.

	print(X_test[0].shape)

	#flattening 
	X_train = X_train.reshape((X_train.shape[0], np.prod(X_train.shape[1:])))
	X_val = X_val.reshape((X_val.shape[0], np.prod(X_val.shape[1:])))
	X_test = X_test.reshape((X_test.shape[0], np.prod(X_test.shape[1:])))

	selected_labels = range(num_of_lables)

	#one-hot encoding
	Y_train = np_utils.to_categorical(y_train-1)
	Y_val = np_utils.to_categorical(y_val-1)
	Y_test = np_utils.to_categorical(y_test-1)


	input_dims = np.prod(X_test.shape[1:]) #784
	nb_classes = Y_train.shape[1]




	
	activation = LeakyReLU(alpha=0.01)
	optimizer = Adam(learning_rate=1e-03)
	regularizer = None#L2(1e-03)
	initializer = HeNormal(seed=SEED)
	loss = 'categorical_crossentropy'
	dropout = 0.1




	model = build_model((input_dims,), nb_classes, activation, initializer, regularizer, dropout)

	model.load_weights('reg/acc2.hdf5')

	activation = LeakyReLU(alpha=0.01)
	optimizer = Adam(learning_rate=1e-03)
	initializer = HeNormal(seed=SEED)
	loss = 'categorical_crossentropy'

	model.compile(optimizer=optimizer, loss=loss, metrics=['categorical_accuracy'])


	score, acc = model.evaluate(X_test, Y_test,
                            batch_size=256)
	print('Test score:', score)
	print('Test accuracy:', acc)


	Y_val_pred = model.predict(X_val)
	y_val_pred = Y_val_pred.argmax(1)
	cm = confusion_matrix(y_val, y_val_pred+1)
	fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(20,20))
	plt.show()

	viridis = matplotlib.cm.get_cmap('viridis_r', 330)
	newcolors = viridis(np.linspace(0, 1, 330))
	white = np.array([1, 1, 1, 1])
	newcolors[:10, :] = white
	newcmp = ListedColormap(newcolors)

	Y_test_pred = model.predict(X_test)
	y_test_pred = Y_test_pred.argmax(1)
	cm = confusion_matrix(y_test, y_test_pred+1)
	fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(20,20), colorbar=True, cmap=newcmp)
	plt.xticks(np.arange(50), np.arange(1, 51))
	plt.yticks(np.arange(50), np.arange(1, 51))
	plt.show()


def format(s, activation, loss, optimizer, e, bs):
	return s.format(activation if isinstance(activation, str) else type(activation).__name__,
					loss, 
					type(optimizer).__name__, 
					e, 
					bs)


if __name__ == '__main__':
	start()
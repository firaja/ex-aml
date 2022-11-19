#!/usr/bin/env python

"""assignment2.py: Solution to the second assignment (autoencoder)."""
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



def build_autoencoder(input_shape, encoding_dim, activation, dropout, regularizer):
	input_ = Input(shape=input_shape)

	encoder_1 = Dense(256, activation = activation, kernel_regularizer=regularizer, name = "downsampling_hidden_2")(input_)
	d1 = Dropout(dropout)(encoder_1)
	encoder_hidden_2 = Dense(128, activation = activation, kernel_regularizer=regularizer, name = "downsampling_hidden_3")(d1)

	encoded = Dense(encoding_dim, activation=activation, kernel_regularizer=regularizer, name = "latent")(encoder_hidden_2)

	decoder_1 = Dense(128, activation = activation, kernel_regularizer=regularizer, name = "upsampling_hidden_1")(encoded)
	d2 = Dropout(dropout)(decoder_1)
	decoder_2 = Dense(256, activation = activation, kernel_regularizer=regularizer, name = "upsampling_hidden_2")(d2)
	d3 = Dropout(dropout)(decoder_2)
	decoder_3 = Dense(512, activation = activation, kernel_regularizer=regularizer, name = "upsampling_hidden_3")(d3)
	
	decoded = Dense(28*39, activation='sigmoid', name = "decoder")(decoder_3)

	autoencoder = Model(input_, decoded)

	autoencoder.summary()

	return input_, encoded, decoded, autoencoder


def start():
	X_train, y_train, X_test, y_test = load_data()

	
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

	#one-hot encoding
	Y_train = np_utils.to_categorical(y_train)
	Y_val = np_utils.to_categorical(y_val)
	Y_test = np_utils.to_categorical(y_test)


	input_dims = np.prod(X_test.shape[1:]) #784
	nb_classes = Y_train.shape[1]




	# configurations
	activation = 'sigmoid'
	optimizer = Adam(learning_rate=1e-03)
	regularizer = None
	initializer = HeNormal(seed=SEED)
	loss = 'binary_crossentropy'
	dropout = 0.1

	




	# Plot training & validation accuracy values
	plt.plot(history.history['mse'])
	plt.plot(history.history['val_mse'])
	plt.title('Model MSE')
	plt.ylabel('MSE')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	#plt.savefig('./images/auto/accuracy.png')

	plt.clf()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper right')
	#plt.savefig('./images/auto/loss.png')



def format(s, activation, loss, optimizer, e, bs):
	return s.format(activation if isinstance(activation, str) else type(activation).__name__,
					loss, 
					type(optimizer).__name__, 
					e, 
					bs)


if __name__ == '__main__':
	start()
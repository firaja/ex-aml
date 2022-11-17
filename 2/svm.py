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
import random

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
from collections import Counter
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score






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

	#downsampling_hidden_1 = Dense(512, activation = activation, name = "downsampling_hidden_1")(input_)
	downsampling_hidden_2 = Dense(256, activation = activation, kernel_regularizer=regularizer, name = "downsampling_hidden_2")(input_)
	d1 = Dropout(dropout)(downsampling_hidden_2)
	downsampling_hidden_3 = Dense(128, activation = activation, kernel_regularizer=regularizer, name = "downsampling_hidden_3")(d1)

	encoded = Dense(encoding_dim, activation=activation, kernel_regularizer=regularizer, name = "latent")(downsampling_hidden_3)

	upsampling_hidden_1 = Dense(128, activation = activation, kernel_regularizer=regularizer, name = "upsampling_hidden_1")(encoded)
	d2 = Dropout(dropout)(upsampling_hidden_1)
	upsampling_hidden_2 = Dense(256, activation = activation, kernel_regularizer=regularizer, name = "upsampling_hidden_2")(d2)
	d3 = Dropout(dropout)(upsampling_hidden_2)
	upsampling_hidden_3 = Dense(512, activation = activation, kernel_regularizer=regularizer, name = "upsampling_hidden_3")(d3)
	
	decoded = Dense(28*39, activation='sigmoid', name = "decoder")(upsampling_hidden_3)

	autoencoder = Model(input_, decoded)

	autoencoder.summary()

	return input_, encoded, decoded, autoencoder



def start():
	X_train, y_train, X_test, y_test = load_data()

	
	# Separate the test data
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=SEED, shuffle=True, stratify = None)

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




	
	activation = LeakyReLU(alpha=0.1)
	optimizer = Adam(lr=0.01)
	regularizer = None#L2(1e-05)
	initializer = HeNormal(seed=SEED)
	loss = 'binary_crossentropy'
	dropout = 0.0

	latent_size = 28*39//20


	input_, encoded, decoded, autoencoder = build_autoencoder((input_dims,), latent_size, activation, dropout, regularizer)
	autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['mse'])


	e = 20
	bs = 512


	history = autoencoder.fit(X_train,X_train,
						epochs=e, 
						batch_size=bs, 
						validation_data = (X_val, X_val), 
						callbacks = [])


	
	X_train_encoded = Model(input_, encoded).predict(X_train)
	X_test_encoded = Model(input_, encoded).predict(X_test)



	svm = SVC()
	svm.fit(X_train_encoded, y_train)

	pred = svm.predict(X_test_encoded)

	print(accuracy_score(y_test, pred))


	cm = confusion_matrix(y_test, pred)
	fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(20,20))
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
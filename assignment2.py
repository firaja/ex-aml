#!/usr/bin/env python

"""assignment2.py: Solution to the second assignment."""
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
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import np_utils



BASE_URL = 'https://github.com/DBertazioli/multi-mnist_custom/raw/master/final_dataset/'
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)


def loadData():
	X_train_path = tf.keras.utils.get_file('X_train.npz', BASE_URL + 'X_train.npz')
	y_train_path = tf.keras.utils.get_file('y_train.npz', BASE_URL + 'y_train.npz')
	X_test_path = tf.keras.utils.get_file('X_test.npz', BASE_URL + 'X_test.npz')
	y_test_path = tf.keras.utils.get_file('y_test.npz', BASE_URL + 'y_test.npz')
	with np.load(X_train_path) as X_train, np.load(y_train_path) as y_train, np.load(X_test_path) as X_test, np.load(y_test_path) as y_test:
		return X_train["arr_0"], y_train["arr_0"], X_test["arr_0"], y_test["arr_0"]

def start():
	X_train, y_train, X_test, y_test = loadData()
	
	
	# Separate the test data
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=SEED, shuffle=True, stratify = None)

	#scaling
	X_train = X_train.astype('float32') / 255.
	X_val = X_val.astype('float32') / 255.
	X_test = X_test.astype('float32') / 255.

	#flattening 
	X_train = X_train.reshape((X_train.shape[0], np.prod(X_train.shape[1:])))
	X_val = X_val.reshape((X_val.shape[0], np.prod(X_val.shape[1:])))
	X_test = X_test.reshape((X_test.shape[0], np.prod(X_test.shape[1:])))

	selected_labels = range(100)

	#one-hot encoding
	Y_train = np_utils.to_categorical(y_train, len(selected_labels))
	Y_val = np_utils.to_categorical(y_val, len(selected_labels))
	Y_test = np_utils.to_categorical(y_test, len(selected_labels))


	input_dims = np.prod(X_test.shape[1:]) #784
	nb_classes = Y_train.shape[1]


	input = Input(shape=(input_dims,)) #28x28
	activation = LeakyReLU(alpha=0.1)

	# 1st hiddent layer
	first_hidden = Dense(1024, activation = activation)(input)

	# 2nd hidden layer
	second_hidden = Dense(512, activation = activation)(first_hidden)

	third_hidden = Dense(128, activation = activation)(second_hidden)

	fourth_hidden = Dense(64, activation = activation)(third_hidden)
	# parallel_path = Dense(blabla)(third_hidden)
	fifth_hidden = Dense(16, activation = activation)(fourth_hidden)
	# output = fifth_hidden + parallel_path
	# output layer
	# FC@nb_classes+softmax 
	output = Dense(nb_classes, activation = "softmax")(fifth_hidden)

	# Model wraps input->output 
	model = Model(input, output)

	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['categorical_accuracy'])
	model.summary()

	# Checkpoints
	mcp_save_acc = ModelCheckpoint('./acc2.hdf5',
									save_best_only=True,
									monitor='val_categorical_accuracy', mode='max')
	mcp_save_loss = ModelCheckpoint('./loss2.hdf5',
									save_best_only=True,
									monitor='val_loss', mode='min')

	n_epochs = 100
	batch_size = 512

	logdir = os.path.join("logs", "basic_model")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

	model.fit(X_train,Y_train, epochs=n_epochs, batch_size=batch_size, validation_data = (X_val, Y_val), callbacks = [tensorboard_callback])


if __name__ == '__main__':
	start()
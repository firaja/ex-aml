#!/usr/bin/env python

"""reg-eval.py: Solution to the second assignment (regularization)."""
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
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(16, (3, 3), activation=activation, kernel_initializer=initializer, padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dropout(dropout))
	model.add(Dense(classes, activation='softmax'))
	return model


def start():
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


	#scaling
	X_train = X_train.astype('float32') / 255.
	X_test = X_test.astype('float32') / 255.

	X_train = np.expand_dims(X_train, -1)
	X_test = np.expand_dims(X_test, -1)

	#one-hot encoding
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)


	# Separate the test data
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=SEED, shuffle=True, stratify = None)
	


	input_dims = X_test.shape[1:]
	nb_classes = y_train.shape[1]


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
	model.compile(optimizer=optimizer, loss=loss, metrics=['categorical_accuracy'])
	model.summary()


	# Checkpoints
	mcp_save_acc = ModelCheckpoint('./acc2.hdf5',
									save_best_only=True,
									monitor='val_categorical_accuracy', 
									mode='max')
	mcp_save_loss = ModelCheckpoint('./loss2.hdf5',
									save_best_only=True,
									monitor='val_loss', 
									mode='min')

	# early stopping
	early_stopping = EarlyStopping(monitor='val_categorical_accuracy', 
									patience=10, 
									min_delta=0.005, 
									mode='max',
									restore_best_weights=True,
									verbose=1)

	

	# trainig
	history = model.fit(X_train, y_train,
						epochs=e, 
						batch_size=bs, 
						validation_data = (X_val, y_val), 
						callbacks = [mcp_save_acc, mcp_save_loss, early_stopping])

	# plot training & validation accuracy values
	plt.plot(history.history['categorical_accuracy'])
	plt.plot(history.history['val_categorical_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(format('./images/accuracy-{}-{}-{}-{}-{}-{}-{}.png', activation, regularizer, loss, optimizer, e, bs, dropout))

	plt.clf()

	# plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper right')
	plt.savefig(format('./images/loss-{}-{}-{}-{}-{}-{}-{}.png', activation, regularizer, loss, optimizer, e, bs, dropout))


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
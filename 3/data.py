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
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from collections import Counter
import statistics



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

	print(len(y_train), len(y_test))

	num = 10
	images = X_train[:num]
	labels = y_train[:num]
	num_row = 2
	num_col = 5
	# plot images
	fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
	for i in range(num):
		ax = axes[i//num_col, i%num_col]
		ax.imshow(images[i], cmap='gray')
		ax.set_title('Number {}'.format(labels[i]))
		ax.set_yticklabels([])
		ax.set_xticklabels([])
	plt.tight_layout()

	plt.show()

	cnt = Counter()
	for x in y_train:
		cnt[x] += 1
	print(cnt)

	mean = statistics.mean(cnt.values())
	stdev = statistics.stdev(cnt.values())

	print('mean: {}    stdev: {}'.format(mean, stdev))

	#plt.style.use('seaborn-dark-palette')

	fig, ax = plt.subplots(figsize=(10,5))
	ax.bar(range(10), np.bincount(y_train), width=0.5, align='center', label='training samples')
	ax.bar(range(10), np.bincount(y_test), width=0.5, align='edge', label='test samples')
	ax.set(xticks=range(10), xlim=[-1, 10], title='Training and test data distribution')
	ax.legend()

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
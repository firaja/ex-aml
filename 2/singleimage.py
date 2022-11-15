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
from collections import Counter
import statistics



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




def start():
	X_train, y_train, X_test, y_test = load_data()

	cnt = Counter()
	for x in y_train:
		cnt[x] += 1
	print(cnt)

	mean = statistics.mean(cnt.values())
	stdev = statistics.stdev(cnt.values())

	print('mean: {}    stdev: {}'.format(mean, stdev))

	n, bins, patches = plt.hist(y_train, bins=50)
	plt.show()


	number = 39

	trains = []
	for i in range(len(X_train)):
		if y_train[i] == number:
			trains.append(X_train[i])

	w = 39
	h = 28
	fig = plt.figure(figsize=(20, 20))
	columns = 5
	rows = 5
	for i in range(1, columns*rows +1):
	    img = trains[i]
	    fig.add_subplot(rows, columns, i)
	    plt.imshow(img)
	plt.show()


def format(s, activation, loss, optimizer, e, bs):
	return s.format(activation if isinstance(activation, str) else type(activation).__name__,
					loss, 
					type(optimizer).__name__, 
					e, 
					bs)


if __name__ == '__main__':
	start()
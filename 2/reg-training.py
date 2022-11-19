#!/usr/bin/env python

"""reg-training.py: Solution to the second assignment (regularized model)."""
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
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.initializers import HeNormal, HeUniform
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from torchvision import transforms
from PIL import Image
from sklearn.utils import shuffle
from skimage.util import random_noise
from scipy.ndimage import shift




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
	model.add(Dense(70, activation = activation,  kernel_initializer=initializer, kernel_regularizer=regularizer))
	model.add(Dropout(dropout))
	model.add(Dense(classes, activation = "softmax", kernel_initializer=initializer, kernel_regularizer=regularizer))
	return model

def augment_data(X_train, y_train, n):
	cnt = Counter()
	for x in y_train:
		cnt[x] += 1
	print(cnt)

	values = cnt.values()

	h = max(values)

	X_res = X_train
	y_res = y_train

	for k in cnt:
		samples = get_samples(k, X_train, y_train)

		x = (h - len(samples)) // 10
		results = rotate(samples, x, random.uniform(-15, 15))
		
		y_res = np.append(y_res, [k]*len(results))
		X_res = np.append(X_res, results, axis=0)


	return X_res, y_res


def rotate(samples, x, rotation):
	result = torch.FloatTensor(x, 28, 39)
	tensor = transforms.ToTensor()
	l = len(samples)

	for i in range(x):
		selector = i
		if i >= l:
			selector = random.randint(0, l-1)
		img = Image.fromarray(samples[selector], mode='L')
		result[i] = tensor(img.rotate(rotation))
	return np.array(result)
		


def add_noise(img):
	gaussian = np.random.normal(0, 0.5, (img.shape[0],img.shape[1])) 
	return img + gaussian


def get_samples(num, X_train, y_train):
	result = []
	for i in range(len(X_train)):
		if y_train[i] == num:
			result.append(X_train[i])
	return result


def shift_image(image, dx, dy):
	image = shift(image, [dy, dx], cval=0, mode="constant")
	return image



def start():
	X_train, y_train, X_test, y_test = load_data()


	# Separate the test data
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=SEED, shuffle=True, stratify = None)

	# data augmentation (disabled)
	# num_of_lables = len(set(np.concatenate((y_train, y_test))))
	# X_train, y_train = augment_data(X_train, y_train, num_of_lables)
	# X_train, y_train = shuffle(X_train, y_train, random_state=SEED)
	
	



	#scaling
	X_train = X_train.astype('float32') / 255.
	X_val = X_val.astype('float32') / 255.
	X_test = X_test.astype('float32') / 255.


	#flattening 
	X_train = X_train.reshape((X_train.shape[0], np.prod(X_train.shape[1:])))
	X_val = X_val.reshape((X_val.shape[0], np.prod(X_val.shape[1:])))
	X_test = X_test.reshape((X_test.shape[0], np.prod(X_test.shape[1:])))


	#one-hot encoding
	Y_train = np_utils.to_categorical(y_train-1)
	Y_val = np_utils.to_categorical(y_val-1)
	Y_test = np_utils.to_categorical(y_test-1)


	input_dims = np.prod(X_test.shape[1:]) #784
	nb_classes = Y_train.shape[1]




	# configurations
	activation = LeakyReLU(alpha=0.01)
	optimizer = Adam(learning_rate=1e-03)
	regularizer = None
	initializer = HeNormal(seed=SEED)
	loss = 'categorical_crossentropy'
	dropout = 0.1
	e = 50
	bs = 256


	# model definition
	model = build_model((input_dims,), nb_classes, activation, initializer, regularizer, dropout)
	model.compile(optimizer=optimizer, loss=loss, metrics=['categorical_accuracy'])
	model.summary()


	# Checkpoints
	mcp_save_acc = ModelCheckpoint('./reg/acc2.hdf5',
									save_best_only=True,
									monitor='val_categorical_accuracy', 
									mode='max')
	mcp_save_loss = ModelCheckpoint('./reg/loss2.hdf5',
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
	history = model.fit(X_train,Y_train,
						epochs=e, 
						batch_size=bs, 
						validation_data = (X_val, Y_val), 
						callbacks = [mcp_save_acc, mcp_save_loss, early_stopping])

	# plot training & validation accuracy values
	plt.plot(history.history['categorical_accuracy'])
	plt.plot(history.history['val_categorical_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(format('./images/reg/accuracy-{}-{}-{}-{}-{}-{}-{}.png', activation, regularizer, loss, optimizer, e, bs, dropout))

	plt.clf()

	# plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper right')
	plt.savefig(format('./images/reg/loss-{}-{}-{}-{}-{}-{}-{}.png', activation, regularizer, loss, optimizer, e, bs, dropout))


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
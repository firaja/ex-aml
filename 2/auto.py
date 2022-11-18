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



def build_autoencoder(input_shape, encoding_dim, activation):
	input = Input(shape=input_shape)

	#downsampling_hidden_1 = Dense(512, activation = activation, name = "downsampling_hidden_1")(input)
	downsampling_hidden_2 = Dense(256, activation = activation, name = "downsampling_hidden_2")(input)
	downsampling_hidden_3 = Dense(128, activation = activation, name = "downsampling_hidden_3")(downsampling_hidden_2)

	encoded = Dense(encoding_dim, activation=activation, name = "latent")(downsampling_hidden_3)

	upsampling_hidden_1 = Dense(128, activation = activation, name = "upsampling_hidden_1")(encoded)
	upsampling_hidden_2 = Dense(256, activation = activation, name = "upsampling_hidden_2")(upsampling_hidden_1)
	upsampling_hidden_3 = Dense(512, activation = activation, name = "upsampling_hidden_3")(upsampling_hidden_2)
	
	decoded = Dense(28*39, activation='sigmoid', name = "decoder")(upsampling_hidden_3)

	autoencoder = Model(input, decoded)

	autoencoder.summary()

	return encoded, decoded, autoencoder


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
	Y_train = np_utils.to_categorical(y_train-1)
	Y_val = np_utils.to_categorical(y_val-1)
	Y_test = np_utils.to_categorical(y_test-1)


	input_dims = np.prod(X_test.shape[1:]) #784
	nb_classes = Y_train.shape[1]




	
	activation = LeakyReLU(alpha=0.01)
	optimizer = Adam(learning_rate=1e-03)
	regularizer = None#L2(1e-03)
	initializer = HeNormal(seed=SEED)
	loss = 'binary_crossentropy'
	dropout = 0.1

	


	d = []


	for cp in range(20, 31):

		latent_size = 28*39//cp

		encoded, decoded, autoencoder = build_autoencoder((input_dims,), latent_size, activation)
		autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['mse'])


		e = 20
		bs = 512


		history = autoencoder.fit(X_train,X_train,
							epochs=e, 
							batch_size=bs, 
							validation_data = (X_val, X_val), 
							callbacks = [])


		score, acc = autoencoder.evaluate(X_test, X_test,
                            batch_size=bs)

		print('MSE:', acc)

		d.append(acc)

	df = pd.DataFrame(d)

	#for k,v in d.items():
	plt.plot(d)
	#plt.legend()
	plt.title('MSE vs compresion factor')
	plt.ylabel('MSE')
	plt.xlabel('Compression factor')
	plt.xticks(np.arange(11), np.arange(20, 31))
	plt.show()


	#fig, axs = plt.subplots(4, 4)
	#rand = X_test[np.random.randint(0, 8000, 16)].reshape((4, 4, 1, 28*39))

	#display.clear_output() # If you imported display from IPython

	#for i in range(4):
	#    for j in range(4):
	#        axs[i, j].imshow(autoencoder.predict(rand[i, j])[0].reshape(28, 39), cmap = "gray")
	#        axs[i, j].axis("off")

	#plt.subplots_adjust(wspace = 0, hspace = 0)
	#plt.show()


	#n = 20
	#random_encodings = np.random.rand(20, latent_size)
	#decoded_imgs = Model(encoded, decoded).predict(random_encodings)

	#plt.figure(figsize=(20, 4))
	#for i in range(n):
	#	# generation
	#	ax = plt.subplot(2, n, i + 1 + n)
	#	plt.imshow(decoded_imgs[i].reshape(28, 39))
	#	plt.gray()
	#	ax.get_xaxis().set_visible(False)
	#	ax.get_yaxis().set_visible(False)
	#plt.show()



def format(s, activation, loss, optimizer, e, bs):
	return s.format(activation if isinstance(activation, str) else type(activation).__name__,
					loss, 
					type(optimizer).__name__, 
					e, 
					bs)


if __name__ == '__main__':
	start()
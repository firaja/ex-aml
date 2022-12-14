#!/usr/bin/env python

"""evaluate1.py: Evaluation of the first assignment."""
__author__      = "David Bertoldi"


import tensorflow as tf
import torch 
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix




SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

class PerformancePlotCallback(tf.keras.callbacks.Callback):
	
	def __init__(self, X_test, y_test, n_epochs_log = 5, save=True):
		self.X_test = X_test
		self.y_test = y_test
		self.n_epochs_log = n_epochs_log
		self.save = save
	
	def on_batch_end(self, epoch, logs={}):
		return
	
	def on_epoch_end(self, epoch, logs={}, ):
		if epoch % self.n_epochs_log == 0:
			plot_decision_regions(self.X_test, self.y_test, clf=self.model, legend=2, colors='C0,C1', markers='oo')
			if self.save:
				plt.savefig('./images/regions/{0}-region.png'.format(epoch), bbox_inches='tight')
			else:
				plt.show()
			plt.close()

def get_samples(m, I, is_grid):
	if not is_grid:
		# Random points
		np.random.seed(1)
		X = np.random.uniform(I[0], I[1], (m, 2))
		X = torch.tensor(X, dtype=torch.float32)
	else:
		# Points on a grid
		n = round(math.sqrt(m))
		grd =  torch.linspace(I[0], I[1], n)
		x1, x2 = torch.meshgrid(grd, grd)
		x1 = x1.flatten().unsqueeze(1)
		x2 = x2.flatten().unsqueeze(1)
		X = torch.cat((x1, x2), dim=1)

	return X

def checkerboard_problem(m=20000, nrows=8, ncols=8, nclasses=2, I=[-1., 1.], is_grid=False):
	"""
	checkerboard problem dataset
	"""
	X = get_samples(m, I, is_grid)
	X_ = X.clone()
	if 0 != I[0] or 1 != I[1]:
		# Map to [0, 1] domain for computing labels
		X_ -= I[0]
		X_.div_(I[1] - I[0])
	x1 = X_[:, 0]
	x2 = X_[:, 1]
	col = (x1 * ncols).long()
	row = (x2 * nrows).long()
	ix = col + row * ncols
	if 0 == ncols % nclasses:
		ix += (row % 2)
	y = ix % nclasses
	# X -= 0.5
	X = torch.tensor(X, dtype=torch.float32)
	y = torch.tensor(y, dtype=torch.uint8)
	return X.numpy(), y.long().numpy()

def plot_data(data, labels, save=True):
	fig, ax = plt.subplots(figsize=(10,10))
	for g in np.unique(labels):
		i = np.where(labels == g)
		ax.scatter(data[i,0], data[i,1], label='Checker ' + str(g), color=['C0', 'C1'][g])
	plt.title("Checkerboard")
	plt.xlabel("X1")
	plt.ylabel("X2")
	plt.legend()
	if save:
		plt.savefig('./images/checkerboard.png')
	else:
		plt.show()
	plt.close()

def preprocess_data(X, scaler):
	scaler.fit(X)
	X = scaler.transform(X)
	return X, scaler

def build_model(input_shape):
	model = Sequential()
	model.add(Dense(200, input_shape=input_shape, activation=LeakyReLU(alpha=0.1), kernel_initializer="he_uniform")) 
	model.add(Dense(150, activation=LeakyReLU(alpha=0.1), kernel_initializer="he_uniform")) 
	model.add(Dense(100, activation=LeakyReLU(alpha=0.1), kernel_initializer="he_uniform")) 
	model.add(Dense(50, activation=LeakyReLU(alpha=0.1), kernel_initializer="he_uniform")) 
	model.add(Dense(1, activation="sigmoid"))
	return model


def start():

	# Main configurations
	m, n, = 4000, 6
	

	# Generate date
	X, y = checkerboard_problem(m=m, nrows=n, ncols=n, nclasses = 2, I = [10,20])

	
	# Separate the test data
	X, X_test, y, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, shuffle=True, stratify = None)

	# Split the remaining data to train and validation
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=SEED, shuffle=True,  stratify = None)

	

	# Train preprocessing
	X_train, scaler = preprocess_data(X_train, StandardScaler())
	X_test, _ = preprocess_data(X_test, StandardScaler())
	X_val, _ = preprocess_data(X_val, StandardScaler())


	plot_data(X, y)

	# NN model
	model = build_model((X_train.shape[1], ))


	model.load_weights('acc.hdf5')


	Y_test_pred = model.predict(X_test)
	
	y_test_pred = [1 * (x[0]>=0.5) for x in Y_test_pred]
	
	cm = confusion_matrix(y_test, y_test_pred)
	fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(10,10))
	plt.savefig('./images/cm.png', bbox_inches='tight')
	
	plot_decision_regions(X_test, y_test, clf=model, legend=2, colors='C0,C1', markers='oo')
	plt.savefig('./images/region.png', bbox_inches='tight')

	plt.close()





if __name__ == '__main__':
	start()
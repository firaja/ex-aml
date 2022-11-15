#!/usr/bin/env python

"""assignment1.py: Solution to the first assignment."""
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LeakyReLU




SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

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




def start():

	# Main configurations
	m, n, = 1000, 20
	loss, opt, e, bs = 'binary_crossentropy', 'adam', 100, 4

	# Generate date
	X, y = checkerboard_problem(m=m, nrows=n, ncols=n, nclasses = 2, I = [10,20])

	unique, counts = np.unique(y, return_counts=True)

	print(dict(zip(unique, counts)))

	
	# Data split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, shuffle = True, stratify = None)

	
	print (np.average(y))
	print (np.var(y))

	print (np.average(X_train))
	print (np.var(X_test))


	# Train preprocessing
	X_train, scaler = preprocess_data(X_train, StandardScaler())
	X_test, _ = preprocess_data(X_test, StandardScaler())

	print (np.average(X_train))
	print (np.var(X_test))

	plot_data(X, y)





if __name__ == '__main__':
	start()
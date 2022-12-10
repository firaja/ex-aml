#!/usr/bin/env python

"""trainer.py: Solution to the fourth assignment."""
__author__      = "David Bertoldi"


import tensorflow as tf
import torch 
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import cv2

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report
from sklearn import decomposition






SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)



CLASSES = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
IMAGE_SIZE=(224, 224)


def parse_arguments():
	parser = argparse.ArgumentParser(description='Assignment 4')

	parser.add_argument('--cut', type=str, const='fc1', default='fc1', nargs='?', choices=['fc1', 'block4_pool', 'block3_pool'], help='Layer to cut')
	parser.add_argument('--c1', type=float, const=100, default=100, nargs='?', help='SVC cost parameter')
	parser.add_argument('--c2', type=float, const=5, default=5, nargs='?', help='SVC cost parameter after PCA')
	parser.add_argument('--skip', default=False, action='store_true', help='Go to PCA')
	parser.add_argument('--train', type=float, const=800, default=800, nargs='?', help='Samples in training set')
	parser.add_argument('--test', type=float, const=300, default=300, nargs='?', help='Samples in test set')
	return parser.parse_args()


def build_model(cut=None):
	base = VGG16(weights='imagenet')
	if cut:
		return Model(inputs=base.input, outputs=base.get_layer(cut).output)
	else:
		return base


def load_imgs(base_dir, n_examples):
	X = []
	Y = []
	
	for class_name in CLASSES:
	
		for img in os.listdir(base_dir + class_name):
			path = os.path.join(base_dir + class_name, img)	
			Y.append(CLASSES.index(class_name))	
			X.append(image.img_to_array(image.load_img(path, target_size=IMAGE_SIZE)))
	
			if (len(Y) % n_examples == 0):
				break

	X = np.array(X)
	Y = np.array(Y)

	idx = np.random.permutation(len(X))
	X, Y = X[idx], Y[idx]
	return X, Y



def predict(svc, features, y, cmap):
	y_pred = svc.predict(features)
	print(classification_report(y_pred, y, target_names=CLASSES))
	cm = confusion_matrix(y_pred, y)
	fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(20,20), colorbar=True, cmap=cmap)
	plt.show()
		


def start():

	args = parse_arguments()
	
	X_train, y_train = load_imgs('./data/seg_train/seg_train/', args.train)
	X_test, y_test = load_imgs('./data/seg_test/seg_test/', args.test)


	X_train = preprocess_input(X_train)
	X_test = preprocess_input(X_test)



	model_cut1 = build_model(cut=args.cut)
	model_cut1.summary()
	

	
	features = model_cut1.predict(X_train)
	
	features_test = model_cut1.predict(X_test)

	if len(features.shape) == 4:
		final_dim = features.shape[1] * features.shape[2] * features.shape[3]
		features = features.reshape((features.shape[0], final_dim))
		features_test = features_test.reshape((features_test.shape[0], final_dim))

	
	n_train, z = features.shape
	n_test, z = features_test.shape
	numFeatures =  z
	x, y = 1, 1
	if len(features.shape) == 4:
		n_train, x, y, z = train_features.shape
		n_test, x, y, z = test_features.shape
		numFeatures = x * y * z

	print('number of features {}'.format(numFeatures))

	del X_train, X_test

	pca = decomposition.PCA(n_components = 2)

	X = features.reshape((n_train, x*y*z))
	pca.fit(X)

	C = pca.transform(X) 
	C1 = C[:,0]
	C2 = C[:,1]


	plt.subplots(figsize=(10,10))

	for i, class_name in enumerate(CLASSES):
		plt.scatter(C1[y_train == i][:1000], C2[y_train == i][:1000], label = class_name, alpha=0.4)
	plt.legend()
	plt.title("PCA Projection")
	plt.show()

	if not args.skip:
		svc = SVC(C=args.c1, kernel='rbf', gamma='scale')
		svc = svc.fit(features, y_train)

		
		predict(svc, features, y_train, "Blues")

		predict(svc, features_test, y_test, "Greens")

		del svc, pca

		pca = decomposition.PCA(n_components=min(features.shape[0], features.shape[1]), copy=False)
		pca.fit(features)

		plt.figure(figsize=(15, 4))
		cumulative = np.cumsum(pca.explained_variance_ratio_)
		plt.xticks(np.arange(0, len(cumulative)+1, 500))
		plt.plot(cumulative, linewidth=3.0)
		plt.grid()
		plt.xlabel('Number of components')
		plt.ylabel('Cumulative explained variance')
		plt.show()


	pca = decomposition.PCA(n_components=0.9, copy = False)

	features = pca.fit_transform(features)
	print('new features shape:', features.shape)
	
	features_test = pca.transform(features_test)


	svc = SVC(C=args.c2, kernel='rbf', gamma='scale')

	svc = svc.fit(features, y_train)

	predict(svc, features, y_train, "Blues")

	predict(svc, features_test, y_test, "Greens")



	



if __name__ == '__main__':
	start()
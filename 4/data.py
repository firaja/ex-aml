import tensorflow as tf
import torch 
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import time

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
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.preprocessing import image
from scipy.interpolate import make_interp_spline


multiclass = np.array([[1000, 0, 0, 0, 0, 0],
                       [0, 1000, 0, 0, 0, 0],
                       [0, 0, 1000, 0, 0, 0],
                       [0, 0, 0, 1000, 0, 0],
                       [0, 0, 0, 0, 1000, 0],
                       [0, 0, 0, 0, 0, 1000]])



class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']

fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                figsize=(20,20),
                                class_names=class_names, cmap='Blues')
plt.show()
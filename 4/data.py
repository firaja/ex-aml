import tensorflow as tf
import torch 
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import cv2
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
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.preprocessing import image
from scipy.interpolate import make_interp_spline


x1_auto = [0.335, 0.369, 0.353, 0.341, 0.362]
x1_auto = np.average(x1_auto)
x1_scale = [0.281, 0.285, 0.269, 0.293, 0.305]
x1_scale = np.average(x1_scale)

x10_auto = [0.474, 0.472, 0.472, 0.456, 0.498]
x10_auto = np.average(x10_auto)
x10_scale = [0.419, 0.442, 0.419, 0.411, 0.429]
x10_scale = np.average(x10_scale)

x100_auto = [0.562, 0.573, 0.550, 0.559, 0.569]
x100_auto = np.average(x100_auto)
x100_scale = [0.520, 0.521, 0.523, 0.511, 0.532]
x100_scale = np.average(x100_scale)


auto = [x1_auto, x10_auto, 0.5, x100_auto]

#plt.plot([1, 10, 20,  100], auto, 'r', label="auto") 
#plt.plot([1, 10, 20, 100], [x1_scale, x10_scale, 0.45,  x100_scale], 'b', label="scale")
#plt.legend(loc="lower right")
#plt.xlabel('cost')
#plt.ylabel('score')
#plt.show()

train = (np.average([2191,2271,2404,2512,2274,2382]), np.std([2191,2271,2404,2512,2274,2382]))
test = (np.average([437, 474, 553,525,510,501]), np.std([437, 474,553,525,510,501]))


print(train, test)
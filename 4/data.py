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


m = ["fc1", "block4_pool", "block3_pool"]
x = np.arange(len(m))

df = {'train': [0.99, 0.99, 0.99], 'test': [0.88, 0.86, 0.77]}

plt.bar(x - 0.2, df["train"], width = 0.35, label = 'Train')
plt.bar(x + 0.2, df["test"], width = 0.35, label = 'Test')

for i in range(3):
    plt.text(x = x[i]-0.3 , y = df["train"][i]+0.04, s = "%.2f" % df["train"][i], size = 10)
    plt.text(x = x[i]+0.1 , y = df["test"][i]+0.04, s = "%.2f" % df["test"][i], size = 10)
  
plt.xticks(x, m)
plt.ylim(top = 1.1)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.legend(loc = 'lower left', facecolor='white', framealpha=1)


plt.subplots_adjust(bottom= 0.2, top = 0.98)
plt.show()
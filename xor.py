# Imports

#Deep learning 
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


#essentials
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#toy data generation
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

#decision boundaries easy peasy
from mlxtend.plotting import plot_decision_regions

#feel free to adjust this param to fit your own screen
# plt.rcParams["figure.figsize"]=(10,10)

#A utility function to plot our generated data
def plot_data(data, labels, title = "Checkerboard"):
    # Generate scatter plot for training data
    fig, ax = plt.subplots(figsize=(10,10))
    for g in np.unique(labels):
        i = np.where(labels == g)
        ax.scatter(data[i,0], data[i,1], label='Checker ' + str(g))
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    
    plt.show()

def preprocess_labels(labels, encoder=None, categorical=True):
    """Encode labels with values among 0 and `n-classes-1`"""
    if not encoder:
        encoder = LabelEncoder()
    encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


#a custom callback in keras -- util to visualize the decision regions
class PerformancePlotCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, X_test, y_test, n_epochs_log = 5):
        self.X_test = X_test
        self.y_test = y_test
        self.n_epochs_log = n_epochs_log
    def on_batch_end(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}, ):
        if epoch % self.n_epochs_log == 0:
        
          plot_decision_regions(self.X_test, self.y_test, clf=self.model, legend=2)
          plt.show()
          plt.close()

num_samples_total=1000

X, y = make_blobs(n_samples = num_samples_total, centers = [(0,0), (1,1), (0, 1), (1,0)], n_features = 2, center_box=(0, 1), cluster_std = 0.05)

plot_data(X, y)

y_enc, encoder = preprocess_labels(y)

#split the data and prepare for a general ML training 
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.33, random_state=42, shuffle = True, stratify = y_enc)

nb_classes = y_train.shape[1]
print(nb_classes, "classes")

input_dims = X_train.shape[1]
print(input_dims, "dims")

show_boundaries = PerformancePlotCallback(X_test, y_test, n_epochs_log = 5)

model = Sequential()
model.add(Dense(10, input_shape=(input_dims,)))
model.add(Activation("gelu"))
model.add(Dense(nb_classes))
model.add(Activation("softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics = ["accuracy"])
model.fit(X_train, y_train, epochs=26, batch_size=16, verbose=1, validation_split=0.2)
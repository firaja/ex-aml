#SEED ALL for reproducibility
from numpy.random import seed
import tensorflow
import torch #don't mind some torch operations -- just use the functions as-is :)

#please do not change the seed
cherrypicked_seed = 42

seed(cherrypicked_seed)
tensorflow.random.set_seed(cherrypicked_seed)
torch.manual_seed(cherrypicked_seed)

#essentials
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


#%matplotlib inline

import math

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


X, y = checkerboard_problem(m = 4000, nrows = 6, ncols = 6, nclasses = 2, I = [10,20])

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

#plot_data(X, y)



#split the data and prepare for a general ML training 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=cherrypicked_seed, shuffle = True, stratify = None)


# Set the input shape
feature_vector_shape = X_train.shape[1]
input_shape = (feature_vector_shape,)

#just pick a single label to showcase the dimension
print(X_train.shape[1])
#simplest case for output-dimension --> binary classification w/o one-hot encoding
output_dimension = 1 

feature_vector_shape, input_shape, output_dimension

# Create the model
#Sequential API -- basically a container
model = Sequential()

#the first hidden layer (note -- input layer gets created automatically, once input_shape is specified)
hidden_layer = Dense(10, input_shape=input_shape, activation="relu")
#add it to the sequential stack
model.add(hidden_layer)



#design an output layer -- with just another way of specifying the activation
output_layer = Dense(output_dimension, activation="softmax")
#add it to the sequential stack
model.add(output_layer)

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, validation_split=0.2)

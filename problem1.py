#SEED ALL for reproducibility
from numpy.random import seed
import tensorflow
import torch 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from mlxtend.plotting import plot_decision_regions
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import LeakyReLU

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



import math

class PerformancePlotCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, X_test, y_test, n_epochs_log = 5):
        self.X_test = X_test
        self.y_test = y_test
        self.n_epochs_log = n_epochs_log
    def on_batch_end(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}, ):
        if epoch % self.n_epochs_log == 0:
        
          plot_decision_regions(self.X_test, self.y_test, clf=self.model, legend=2, colors='C1,C0', markers='oo')
          #plt.show()
          plt.savefig(str(epoch) + '-region.png')
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


X, y = checkerboard_problem(m = 4000, nrows = 6, ncols = 6, nclasses = 2, I = [10,20])


#A utility function to plot our generated data
def plot_data(data, labels, title = "Checkerboard"):
    # Generate scatter plot for training data
    fig, ax = plt.subplots(figsize=(10,10))
    for g in np.unique(labels):
        i = np.where(labels == g)
        ax.scatter(data[i,0], data[i,1], label='Checker ' + str(g), color=['C0', 'C1'][g])
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
        y = tensorflow.keras.utils.to_categorical(y)
    return y, encoder

def preprocess_data(X, scaler=None):
    """Preprocess input data by standardise features 
    by removing the mean and scaling to unit variance"""
    if not scaler:
        scaler = StandardScaler()
    
    scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler
#y_enc, encoder = preprocess_labels(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle = True, stratify = None)
print(y_train)
scaler = MinMaxScaler()
X_train, scaler = preprocess_data(X_train, scaler = StandardScaler())

#notice how we use on the test set the SAME scaler fitted on training set.
X_test, scaler = preprocess_data(X_test, scaler = StandardScaler())



#split the data and prepare for a general ML training 

plot_data(X, y)

feature_vector_shape = X_train.shape[1]
input_shape = (feature_vector_shape,)


input_dims = X_train.shape[1]
print(input_dims, "dims")


model = Sequential()
model.add(Dense(200, input_shape=input_shape, activation=LeakyReLU(alpha=0.1), kernel_initializer="he_uniform")) 

model.add(Dense(150,  activation=LeakyReLU(alpha=0.1), kernel_initializer="he_uniform")) 
model.add(Dense(100,  activation=LeakyReLU(alpha=0.1), kernel_initializer="he_uniform")) 
model.add(Dense(50,  activation=LeakyReLU(alpha=0.1), kernel_initializer="he_uniform")) 

model.add(Dense(1, activation="sigmoid"))

show_boundaries = PerformancePlotCallback(X_test, y_test, n_epochs_log = 10)
# Configure the model and start training
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, validation_split=0.2, callbacks=[show_boundaries])

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('prec.png')

plt.clf()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig('loss.png')

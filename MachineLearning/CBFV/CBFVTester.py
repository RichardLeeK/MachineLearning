import pickle
import keras.backend as k
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar 
import numpy as np
import pickle
import datetime
from ml.public import *
from keras.utils import np_utils

def lstm_data_gen(X_train, Y_train, cnt):
  n_X_train = []
  cur_X_train = []
  n_Y_train = []
  cur_Y_train = []
  for i in range(cnt, len(X_train)):
    n_X_train.append(X_train[i-cnt:i])
    n_Y_train.append(Y_train[i])
  return np.array(n_X_train), np.array(n_Y_train)


mode = 'PI_REG_'
file = open('data/dataset_PI_reg.pickle', 'rb')
data = pickle.load(file)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

data_dim = len(X_train[0])
timesteps = 8
num_classes = 2

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))
model.add(LSTM(16, return_sequences=True, input_shape=(timesteps, data_dim)))
model.add(LSTM(8))
model.add(Dropout(0.9))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])


X_tr, y_tr = lstm_data_gen(X_train, y_train, timesteps)
X_te, y_te = lstm_data_gen(X_test, y_test, timesteps)
y_tr = np_utils.to_categorical(y_tr)
y_te = np_utils.to_categorical(y_te)
model.fit(X_tr, y_tr, batch_size=64, epochs=10, validation_data=(X_te, y_te))
Y_pred = model.predict(X_te)

pen = open('res/' + mode + '.csv', 'w')




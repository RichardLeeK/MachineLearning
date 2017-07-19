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

def ann_build_generator(latent_size):
  model = Sequential()
  model.add(Dense(output_dim=5, input_dim=latent_size, activation='sigmoid', init='glorot_uniform'))
  model.add(Dense(output_dim=5, activation='sigmoid', init='glorot_uniform'))
  model.add(Dense(output_dim=1, activation='sigmoid', init='glorot_uniform'))
  return model

def ann_train_test(X_train, X_test, Y_train, Y_test):
  st = datetime.datetime.now()
  model = ann_build_generator(len(X_train[0]))
  model.fit(X_train, Y_train, epochs=100)
  with open('network/ann.net', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
  Y_pred = model.predict(X_test)
  tp, tn, fp, fn = pred_test_lstm(Y_pred, Y_test)
  ed = datetime.datetime.now()
  print('Artificial Neural Network finished ' + str(ed - st))
  print('TP: ' + str(tp) + '\tTN: ' + str(tn) + '\tFP: ' + str(fp) + '\tFN: ' + str(fn))
  return 'Artificial Neural Network,' + gen_result_line(tp, tn, fp, fn)

def dnn_build_generator(latent_size):
  model = Sequential()
  model.add(Dense(20, input_dim=latent_size, activation='relu'))
  model.add(Dense(20, input_dim=20, activation='relu'))
  model.add(Dense(10, input_dim=20, activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(10, input_dim=10, activation='relu'))
  model.add(Dense(10, input_dim=10, activation='relu'))
  model.add(Dense(4, input_dim=10, activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(1, input_dim=4, activation='linear'))
  model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])
  return model

def dnn_train_test(X_train, X_test, Y_train, Y_test):
  st = datetime.datetime.now()
  model = dnn_build_generator(len(X_train[0]))
  model.fit(X_train, Y_train, epochs=10)
  with open('network/dnn.net', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
  Y_pred = model.predict(X_test)
  tp, tn, fp, fn = pred_test_lstm(Y_pred, Y_test)
  ed = datetime.datetime.now()
  print('Deep Neural Network finished ' + str(ed - st))
  print('TP: ' + str(tp) + '\tTN: ' + str(tn) + '\tFP: ' + str(fp) + '\tFN: ' + str(fn))
  return 'Deep Neural Network,' + gen_result_line(tp, tn, fp, fn)

def lstm_build_generator(latent_size):
  model = Sequential()
  model.add(LSTM(4, input_shape=(60, latent_size)))
  model.add(Dense(1))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
  return model

def lstm_build_generator2(latent_size):
  model = Sequential()
  model.add(LSTM(4, input_shape=(240, latent_size)))
  model.add(Dense(1))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
  return model

def lstm_data_gen(X_train, Y_train, cnt):
  n_X_train = []
  cur_X_train = []
  n_Y_train = []
  cur_Y_train = []
  for i in range(cnt, len(X_train)):
    n_X_train.append(X_train[i-cnt:i])
    n_Y_train.append(Y_train[i])
  return np.array(n_X_train), np.array(n_Y_train)

def lstm_train_test(X_train, X_test, Y_train, Y_test):
  st = datetime.datetime.now()
  model = lstm_build_generator(len(X_train[0]))
  n_X_train, n_Y_train = lstm_data_gen(X_train, Y_train, 60)
  model.fit(n_X_train, n_Y_train, epochs=50, verbose=2)
  n_X_test, n_Y_test = lstm_data_gen(X_test, Y_test, 60)
  Y_pred = model.predict(n_X_test)
  tp, tn, fp, fn = pred_test_lstm(Y_pred, n_Y_test)

  return 'LSTM,' + gen_result_line(tp, tn, fp, fn)
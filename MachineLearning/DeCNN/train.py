import os
os.environ['KERAS_BACKEND'] = 'theano'

from model import FullyConvolutionalNetwork, PatchBasedCNN
from preprocess import *

import argparse
import h5py

from keras.optimizers import Adam
from keras import backend as K

def crossentropy(y_true, y_pred):
  return -K.sum(y_true*K.log(y_pred))

img_size = 512
patch_size = 8
nb_class = 21
path_to_train = 'train/'
path_to_target = 'target/'
path_to_txt = 'other/train0712_p.txt'

with open(path_to_txt, 'r') as f:
  ls = f.readlines()
names = [l.rstrip('\n') for l in ls]
nb_data = len(names)

FCN = PatchBasedCNN(img_height=patch_size, img_width=patch_size, nb_class=nb_class)
train_model = FCN.create_model()

train_model.fit_generator(generate_train_list(names, path_to_train, path_to_target, img_size, patch_size, nb_class), samples_per_epoch=nb_data, nb_epoch=100)

train_model.save_weights('other/weights')

f = h5py.File('other/weights')

layer_names = [name for name in f.attrs['layer_names']]
fcn = FCN.create_model(train_flag=False)

for i, layer in enumerate(fcn.layers):
  g = f[layer_names[i]]
  weights = [g[name] for name in g.attrs['weight_names']]
  layer.set_weights(weights)
fcn.save_weights('weight/fcn_params_np', overwrite=True)
f.close()

print('saved weights')
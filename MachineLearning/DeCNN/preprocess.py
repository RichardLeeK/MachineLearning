import numpy as np
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from sklearn.feature_extraction.image import extract_patches_2d

def most_common(dat):
  map = {}
  for d in dat:
    if d in map:
      map[d] += 1
    else:
      map[d] = 1
  max_v = 0
  max_k = 0
  for k, v in map.items():
    if v > max_v:
      max_v = v
      max_k = k
  return max_k

def binarylab(labels, size, nb_class):
    y = np.zeros((size,size,nb_class))
    for i in range(size):
        for j in range(size):
            y[i, j, labels[i][j]] = 1
    return y

def load_data(path, size=512, mode=None):
    img = Image.open(path)
    w,h = img.size
    if w < h:
        if w < size:
            img = img.resize((size, size*h//w))
            w, h = img.size
    else:
        if h < size:
            img = img.resize((size*w//h, size))
            w, h = img.size
    img = img.crop((int((w-size)*0.5), int((h-size)*0.5), int((w+size)*0.5), int((h+size)*0.5)))
    if mode=="original":
        return img

    if mode=="label":
        y = np.array(img, dtype=np.int32)
        mask = y == 255
        y[mask] = 0
        y = binarylab(y, size, 21)
        y = np.expand_dims(y, axis=0)
        return y
    if mode=="data":
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        X = preprocess_input(X)
        return X

def generate_arrays_from_file(names, path_to_train, path_to_target, img_size, nb_class):
    while True:
        for name in names:
            Xpath = path_to_train + "{}.jpg".format(name)
            ypath = path_to_target + "{}.png".format(name)
            X = load_data(Xpath, img_size, mode="data")
            y = load_data(ypath, img_size, mode="label")
            yield (X, y)


def generate_train_list(names, path_to_train, path_to_target, img_size, patch_size, nb_class):
  while True:
    for name in names:
      Xpath = path_to_train + '{}.jpg'.format(name)
      Ypath = path_to_target + '{}.png'.format(name)
      X = load_data(Xpath, img_size, mode='data')
      Y = load_data(Ypath, img_size, mode='label')
      window_shape = (patch_size, patch_size)
      X_patches = extract_patches_2d(X.reshape(img_size, img_size, 3), window_shape)
      Y_patches = extract_patches_2d(Y.reshape(img_size, img_size, 21), window_shape)
      Y_labels = []
      for patch in Y_patches:
        patch_list = patch.reshape(patch_size*patch_size, nb_class)
        most_finder = []
        for p in patch_list:
          label = np.where(p == 1)[0][0]
          most_finder.append(label)
        Y_labels.append(most_common(most_finder))
      Y_labels = np.array(Y_labels).transpose()
      Y_label_gen = np.zeros((len(Y_labels), 21))
      for i in range(len(Y_labels)):
        Y_label_gen[i][int(Y_labels[i])] = 1
      yield (X_patches, Y_label_gen)
def generate_train_data_1(name, path_to_train, path_to_target, img_size, patch_size, nb_class):
  Xpath = path_to_train + '{}.jpg'.format(name)
  Ypath = path_to_target + '{}.png'.format(name)
  X = load_data(Xpath, img_size, mode='data')
  Y = load_data(Ypath, img_size, mode='label')
  window_shape = (patch_size, patch_size)
  X_patches = extract_patches_2d(X.reshape(img_size, img_size, 3), window_shape)
  Y_patches = extract_patches_2d(Y.reshape(img_size, img_size, 21), window_shape)
  Y_labels = []
  for patch in Y_patches:
    patch_list = patch.reshape(patch_size*patch_size, nb_class)
    most_finder = []
    for p in patch_list:
      label = np.where(p == 1)[0][0]
      most_finder.append(label)
    Y_labels.append(most_common(most_finder))
  Y_labels = np.array(Y_labels).transpose()
  Y_label_gen = np.zeros((len(Y_labels), 21))
  for i in range(len(Y_labels)):
    Y_label_gen[i][int(Y_labels[i])] = 1
  return X_patches, Y_label_gen

  """
  Xs = []
  Ys = []
  for name in names:
    Xpath = path_to_train + '{}.jpg'.format(name)
    Ypath = path_to_target + '{}.png'.format(name)
    X = load_data(Xpath, img_size, mode='data')
    Y = load_data(Ypath, img_size, mode='label')
    window_shape = (patch_size, patch_size)
    X_patches = extract_patches_2d(X.reshape(img_size, img_size, 3), window_shape)
    Y_patches = extract_patches_2d(Y.reshape(img_size, img_size, 21), window_shape)
    Y_labels = []
    from statistics import mode
    for patch in Y_patches:
      patch_list = patch.reshape(patch_size*patch_size*nb_class)
      Y_labels.append(mode(patch_list))
    Xs += X_patches
    Ys ++Y_labels)
  return Xs, Ys
  """


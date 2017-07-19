from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

def signal_to_img(signals):
  imgs = []
  for sig in signals:
    dim = len(sig)
    img = np.zeros((dim, dim))
    for i in range(dim):
      idx = round(sig[i] * (dim-1))
      img[dim-idx-1][i] = 255
      if idx > 0:
        img[dim-idx][i] = 125
      if idx < dim-1:
        img[dim-idx-2][i] = 125
    imgs.append(img)
  return np.array(imgs)

def autoencoding(train_x, test_x, img_dim=128, encoding_dim=32):
  input_img = Input(shape=(img_dim**2,))
  encoded = Dense(encoding_dim, activation='relu')(input_img)
  decoded = Dense(img_dim**2, activation='sigmoid')(encoded)
  autoencoder = Model(input_img, decoded)
  encoder = Model(input_img, encoded)
  encoded_input = Input(shape=(encoding_dim,))
  decoder_layer = autoencoder.layers[-1]
  decoder = Model(encoded_input, decoder_layer(encoded_input))
  autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

  r_train_x = train_x.astype('float32')/255
  r_train_x = r_train_x.reshape((len(r_train_x), np.prod(r_train_x.shape[1:])))
   
  r_test_x = test_x.astype('float32')/255
  r_test_x = r_test_x.reshape((len(r_test_x), np.prod(r_test_x.shape[1:])))

  autoencoder.fit(r_train_x, r_train_x, epochs=100, batch_size=100, shuffle=True)
  encoded_imgs = encoder.predict(r_test_x)
  decoded_imgs = decoder.predict(encoded_imgs)
  return decoded_imgs

def autoencoding_cnn(train_x, test_x, img_dim=128, encoding_dim=32):
  input_img = Input(shape=(img_dim, img_dim, 1))
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  encoded = MaxPooling2D((2, 2), padding='same')(x)

  # at this point the representation is (7, 7, 32)

  x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

  autoencoder = Model(input_img, decoded)
  autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

  r_train_x = np.array(train_x).astype('float32')/255
  r_train_x = np.reshape(r_train_x, (len(r_train_x), img_dim, img_dim, 1))

  r_test_x = np.array(test_x).astype('float32')/255
  r_test_x = np.reshape(r_test_x, (len(r_test_x), img_dim, img_dim, 1))

  autoencoder.fit(r_train_x, r_train_x, epochs=100, batch_size=100, shuffle=True)
  decoded_imgs = autoencoder.predict(r_test_x)
  return decoded_imgs



def show_imgs(first, second, img_dim=128, n=8):
  plt.figure(figsize=(15,5))
  for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(first[i].reshape(img_dim, img_dim))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(second[i].reshape(img_dim, img_dim))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()
  return

def load_data(path):
  import os
  files = os.listdir(path)
  signals = []
  for file in files:
    f = open(path+file)
    lines = f.readlines()
    f.close()
    cur_file_sig = []
    for line in lines:
      sl = line.split(',')
      cur_sig = []
      for v in sl[1:]:
        cur_sig.append(float(v))
      cur_file_sig.append(cur_sig)
    signals.append(cur_file_sig)
  return signals, files

def load_data_raw(path):
  import os
  files = os.listdir(path)
  signals = []
  for file in files:
    f = open(path+file)
    lines = f.readlines()
    f.close()
    for line in lines:
      sl = line.split(',')
      cur_sig = []
      for v in sl[1:]:
        cur_sig.append(float(v))
      signals.append(cur_sig)
  max_len = 0
  max_val = 0
  for sig in signals:
    if len(sig) > max_len:
      max_len = len(sig)
    if max(sig) > max_val:
      max_val = max(sig)
  for i in range(len(signals)):
    while True:
      if len(signals[i]) == max_len: break
      signals[i].append(0)
    for j in range(len(signals[i])):
      signals[i][j] /= max_val 
  return signals, files[0]

if __name__=='__main__':
  path = 'D:/Richard/CBFV/Auto-encoder/interpolate/train/'
  signals = load_data_raw(path)
  imgs = signal_to_img(signals)
  rep_imgs = autoencoding(imgs, imgs, img_dim=len(imgs[0]), encoding_dim=32)
  show_imgs(imgs, rep_imgs, img_dim=len(imgs[0]))
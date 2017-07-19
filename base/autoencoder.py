from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

encoding_dim = 32

input_img = Input(shape=(16384,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(16384, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

file = open('D:/Richard/CBFV/Auto-encoder/001040SE_interpolated.csv')
lines = file.readlines()
file.close()
signals = []
for line in lines:
  sl = line.split(',')
  cur_sig = []
  for v in sl[1:]:
    cur_sig.append(float(v))
  signals.append(cur_sig)


imgs = []
for sig in signals:
  dim = len(sig)
  img = np.zeros((dim, dim))
  for i in range(dim):
    idx = round(sig[i] * (dim - 1))
    img[dim-idx-1][i] = 255
    if idx > 0:
      img[dim-idx][i] = 125
    if idx < dim - 1:
      img[dim-idx-2][i] = 125
    
  imgs.append(img)
x = np.array(imgs).astype('float32')/255
x = x.reshape((len(x), np.prod(x.shape[1:])))
autoencoder.fit(x, x, epochs=200, batch_size=100, shuffle=True)
encoded_imgs = encoder.predict(x)
decoded_imgs = decoder.predict(encoded_imgs)
"""
from keras.datasets import mnist
import numpy as np
(x, _), (x2, _) = mnist.load_data()
x = x[:10]
x2 = x2[:10]
x = x.astype('float32')/255
x2 = x2.astype('float32')/255

x = x.reshape((len(x), np.prod(x.shape[1:])))
x2 = x2.reshape((len(x2), np.prod(x2.shape[1:])))

print (x.shape)
print(x2.shape)

autoencoder.fit(x, x, epochs=30, batch_size=100, shuffle=True, validation_data=(x2, x2))

encoded_imgs = encoder.predict(x2)
decoded_imgs = decoder.predict(encoded_imgs)
"""
import matplotlib.pyplot as plt

n = 8

plt.figure(figsize=(20,4))
for i in range(n):
  ax = plt.subplot(2, n, i+1)
  plt.imshow(x[i].reshape(128, 128))
  #plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i].reshape(128, 128))
  #plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.show()



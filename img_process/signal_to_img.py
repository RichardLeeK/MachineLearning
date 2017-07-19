import sys
sys.path.insert(0, 'D:/Sources/Python Source Code')
import numpy as np
from PIL import Image
from scipy.misc import imshow
import base.autoencoder

def interpolated_signal_to_img(signal):
  dim = len(signal)
  img = np.ones((dim, dim))
  for i in range(dim):
    idx = round(signal[i] * (dim-1))
    img[i][idx] = 100
    if idx > 0:
      img[i][idx-1] = 50
    if idx < dim - 1:
      img[i][idx+1] = 50
  im = Image.fromarray(img, 'RGB')
  im.show()
  im.save('img/test.png')
  return img

def signal_to_img(signal):
  dim = len(signal)
  

import os
import numpy as np
from sklearn import manifold


def tSNELearning(data, n_component=3, init='pca'):
  tsne = manifold.TSNE(n_components=n_component, init=init, random_state=0)
  Y = tsne.fit_transform(data)
  return Y

def SpectralEmbeddingLearning(data, n_component=3, n_neighbor=10):
  se = manifold.SpectralEmbedding(n_components=n_component, n_neighbors=n_neighbor)
  Y = se.fit_transform(data)
  return Y

def MDSLearning(data, n_component=3, max_iter=100, n_init=1):
  mds = manifold.MDS(n_components=n_component, max_iter=max_iter, n_init=n_init)
  Y = mds.fit_transform(data)
  return Y

def find_nearest_point(x, y, imgs):
  idx = 0; min_length = 1000000
  for i in range(len(imgs)):
    lenght = (imgs[i][0] - x) ** 2 + (imgs[i][1] - y) ** 2
    if lenght < min_length:
      idx = i
      min_length = lenght
  return idx


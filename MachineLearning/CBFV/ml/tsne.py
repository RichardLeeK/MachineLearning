import sys
sys.path.insert(0, 'D:/Sources/Python Source Code/MachineLearning/CBFV')
import os
import ml.autoencoder as ac
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm


fig_cnt = 5

def gos_loader():
  file = open('GOS.csv')
  lines = file.readlines()
  gos_map = {}
  for line in lines[1:]:
    sl = line.split(',')
    gos_map[int(sl[0])] = int(sl[1])
  return gos_map

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

def kmeans_clustering(X, Y):
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
  pred_Y = kmeans.predict(Y)
  return pred_Y

class ClickEvent:
  def __init__(self, fig, imgs, real_imgs):
    self.fig = plt.get_current_fig_manager().canvas.figure
    self.imgs = imgs
    self.real_imgs = real_imgs
  def onclick(self, event):
    if event.xdata == None or event.ydata == None:
      print('Click correctly!')
      return
    idx = find_nearest_point(event.xdata, event.ydata, self.imgs)
    print(str(idx))
    plt.scatter(self.imgs[idx][0], self.imgs[idx][1], color='r')
    global fig_cnt
    new_fig = plt.figure(fig_cnt)
    fig_cnt += 1
    plt.imshow(self.real_imgs[idx].reshape(128, 128))
    self.fig.canvas.draw()
    plt.show()

def image_print(filename, signals, imgs, rep_imgs, rep_imgs_comp):
  tsne_o = manifold.TSNE(n_components=2, init='pca', random_state=0)
  Y_o = tsne_o.fit_transform(signals)
  tsne_r = manifold.TSNE(n_components=2, init='pca', random_state=0)
  rep_imgs_array = rep_imgs.reshape(len(rep_imgs), 128*128)
  Y_r = tsne_r.fit_transform(rep_imgs_array)
  
  #colors = ['b', 'r', 'g']
  kc = kmeans_clustering(Y_r, Y_r)
  if os.path.isdir('tsne_img/'+filename):
    os.remove('tsne_img/'+filename)
  os.mkdir('tsne_img/'+filename)
  os.mkdir('tsne_img/'+filename+'/ori')
  os.mkdir('tsne_img/'+filename+'/rep')
  os.mkdir('tsne_img/'+filename+'/rep_dnn')
  """
  for i in range(12):
    os.mkdir('tsne_img/'+filename+'/ori/'+str(i))
    os.mkdir('tsne_img/'+filename+'/rep/'+str(i))
    os.mkdir('tsne_img/'+filename+'/rep_dnn/'+str(i))
  """
  for i in range(len(kc)):
    plt.figure()
    plt.imshow(imgs[i].reshape(128, 128))
    plt.savefig('tsne_img/' + filename + '/ori/' + str(i) + '_' + str(round(Y_o[i, 0], 3)) + ',' + str(round(Y_o[i, 1], 3)) + '.png')
    plt.figure()
    plt.imshow(rep_imgs[i].reshape(128, 128))
    plt.savefig('tsne_img/' + filename + '/rep/' + str(i) + '_' + str(round(Y_r[i, 0], 3)) + ',' + str(round(Y_r[i, 1], 3)) + '.png')
    plt.figure()
    plt.imshow(rep_imgs_comp[i].reshape(128, 128))
    plt.savefig('tsne_img/' + filename + '/rep_dnn/' + str(i) + '_' + str(round(Y_r[i, 0], 3)) + ',' + str(round(Y_r[i, 1], 3)) + '.png')
  return

if __name__ ==  '__main__':
  path = 'D:/Richard/CBFV/Auto-encoder/interpolate/test/'
  signals, filenames = ac.load_data(path)
  for i in range(len(filenames)):
    filename = filenames[i].split('_')[0]
    imgs = ac.signal_to_img(signals[i])
    rep_imgs = ac.autoencoding_cnn(imgs, imgs, encoding_dim=32)
    rep_imgs_comp = ac.autoencoding(imgs, imgs, encoding_dim=32)
    image_print(filename, signals[i], imgs, rep_imgs, rep_imgs_comp)
  print('fin')


  """
  fig = plt.figure(1)
  for i in range(len(Y_o)):
    plt.scatter(Y_o[i][0], Y_o[i][1], color=colors[kc[i]])
  ceo = ClickEvent(fig, Y_o, imgs)
  fig.canvas.mpl_connect('button_press_event', ceo.onclick)


  fig2 = plt.figure(2)
  for i in range(len(Y_r)):
    plt.scatter(Y_r[i][0], Y_r[i][1], color=colors[kc[i]])
  cer = ClickEvent(fig2, Y_r, rep_imgs)
  fig2.canvas.mpl_connect('button_press_event', cer.onclick)
  
  plt.show()
  """

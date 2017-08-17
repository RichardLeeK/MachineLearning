from scipy import misc
import matplotlib.pyplot as plt
import manifold as mf
import pickle
import numpy as np
from datetime import datetime
import Optimizer as om

def class_loader():
  file = open('img/label_gos_64_classification.csv')
  lines = file.readlines()
  file.close()
  class_map = {}
  for line in lines[1:]:
    sl = line.split(',')
    class_map[int(sl[0])] = int(sl[1].strip())
  print('class load finish')
  return class_map

def img_loader(ori=True):
  imgs = []
  path = 'ori' if ori else 'rep'
  import os
  files = os.listdir('npy/signal_64/'+path)
  files2 = ['' for i in range(len(files))]
  for f in files:
    files2[int(f.split('.')[0].split('_')[-1])] = f

  cnt = 0
  for i in range(len(files2)):
    #img = misc.imread('npy/signal_64/'+path+'/'+f)
    img = np.load('npy/signal_64/'+path+'/'+files2[i], mmap_mode='r')
    imgs.append(img)
    if cnt % 100 == 0:
      print(str(cnt/100) + '/' + str(len(files2)/100) + ' loading...')
    cnt += 1
  return imgs

if __name__=='__main__':
  cm = class_loader()
  ori = img_loader(True)
  rep = img_loader(False)

  re_ori = np.array(ori).reshape(len(ori), 64*64)
  re_rep = np.array(rep).reshape(len(rep), 64*64)

  perplexities = [5, 10, 15, 20, 25, 30, 50, 100, 200]
  inits = ['random', 'pca']
  pen = open('distances.csv', 'a')
  for i in range(len(perplexities)):
    for j in range(len(inits)):
      cp = datetime.now()
      yo = mf.tSNELearning(re_ori, init='pca')
      cp2 = datetime.now()
      print(cp2 - cp)
      yr = mf.tSNELearning(re_rep, init='pca')
      cp = datetime.now()
      print(cp - cp2)
      gos_color = ['b', 'g', 'r', 'c', 'm', 'y', 'salmon', 'pink', 'gold', 'maroon', 'navy']
      pred_array = om.k_mean_accuracy(yr, cm)
      eds = om.Euclidean_distance(yr, cm)
      sen = str(perplexities[i]) + ',' + inits[j]
      for p in pred_array:
        sen +=  ',' + str(p)
      for e in eds:
        sen += ',' + str(e)
      pen.write(sen + '\n')
      fig1 = plt.figure(1)
      for i in range(len(yo)):
        if i not in cm: break
        plt.scatter(yo[i][0], yo[i][1], color=gos_color[cm[i]])
      plt.savefig(str(perplexities[i]) + '_' + inits[j] + '_ori.png')
      fig2 = plt.figure(2)
      for i in range(len(yr)):
        if i not in cm: break
        plt.scatter(yr[i][0], yr[i][1], color=gos_color[cm[i]])
      plt.savefig(str(perplexities[i]) + '_' + inits[j] + '_rep.png')

      


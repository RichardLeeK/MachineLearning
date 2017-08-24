import numpy as np
from sklearn.cluster import KMeans
import math


def k_mean_accuracy(y, cm, n_cluster=9):
  cluster = KMeans(n_clusters=n_cluster, max_iter=1000)
  cluster.fit(y)
  pred = cluster.predict(y)
  pred_array = np.zeros((9, 9))
  for i in range(len(pred)):
    pred_array[cm[i]][pred[i]] += 1

  accuracy = []
  for i in range(9):
    accuracy.append(np.amax(pred_array[i]) / (np.sum(pred_array[i])+0.00000001))
  return sum(accuracy)/9

def distance(a, b):
  return math.sqrt((a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]))

def Euclidean_distance(y, cm):
  mc = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
  for k, v in cm.items():
    mc[v].append(k)
  ed = []
  for k, v in mc.items():
    val = 0
    cnt = 0
    for f in v:
      for s in v:
        if s > 5000 or f > 5000: continue 
        val += distance(y[f], y[s])
        cnt += 1
    ed.append(val / cnt)
  return ed
     

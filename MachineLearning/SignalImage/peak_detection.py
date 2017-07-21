import numpy as np
import peakutils
import os
import matplotlib.pyplot as plt
from scipy.signal import kaiserord, lfilter, firwin, freqz, find_peaks_cwt

def index_revisor(idxes, sig):
  rev_idxes = list(idxes)
  for i in range(len(idxes) - 1):
    if idxes[i+1] - idxes[i] == 1:
      if sig[idxes[i]] > sig[idxes[i+1]]:
        rev_idxes.remove(idxes[i+1])
      else:
        rev_idxes.remove(idxes[i])
  rev_rev_idxes = rev_idxes.copy()
  for i in range(len(rev_idxes)):
    center = rev_idxes[i]
    if sig[center - 1] > sig[center]:
      rev_rev_idxes.remove(center)
      rev_rev_idxes.append(center-1)
    if sig[center + 1] > sig[center]:
      rev_rev_idxes.remove(center)
      rev_rev_idxes.append(center+1)
  rev_rev_idxes.sort()
  return np.array(rev_rev_idxes)

def peaks_finder(sig):
  indexes = find_peaks_cwt(sig, np.arange(2, 7))
  indexes = index_revisor(indexes, sig)
  indexes = index_revisor(indexes, sig)
  indexes = index_revisor(indexes, sig)
  indexes = index_revisor(indexes, sig)
  return indexes

def peaks_finder_self(sig):
  indexes = []
  for i in range(1, len(sig) - 1):
    if sig[i] >= sig[i-1]-0.005 and sig[i] >= sig[i+1]-0.005:
      indexes.append(i)
  return indexes 

def peaks_finder_combine(sig):
  i1 = peaks_finder(sig)
  i2 = peaks_finder_self(sig)
  indexes = []
  for i in range(128):
    if i in i1 and i in i2:
      indexes.append(i)

  return indexes


def notch_paek_tester(idx, sig):
  sig[idx-10]


def classifier(sig):
  indexes = peaks_finder(sig)
  if len(indexes) == 1:
    return 0

  front_peaks = []
  for i in indexes:
    if i < 68:
      print('a')

  is_second_peak = True if (indexes[1] - indexes[0]) < 22 else False
  if is_second_peak:
    is_notch_peak = True if (indexes[2] - indexes[1]) < 40 else False
    second_peak_idx = indexes[1]
    notch_peak_idx =indexes[2]
  else:
    notch_peak = indexes[1]





f = open('data/interpolate_gos/010010DW_interpolated.csv')
lines = f.readlines()


for line in lines:
  sl = line.split(',')
  cur_sig = []
  cur_x = []
  i = 0 
  for l in sl[1:]:
    cur_sig.append(float(l))  
    cur_x.append(i)
    i  += 1

  fig = plt.figure(1)
  indexes = peaks_finder_combine(cur_sig)
  cur_peak = []
  for idx in indexes:
    cur_peak.append(cur_sig[idx])
  plt.scatter(cur_x, cur_sig)
  plt.scatter(indexes, cur_peak, color='r')


  fig = plt.figure(2)
  indexes = peaks_finder(cur_sig)
  cur_peak = []
  for idx in indexes:
    cur_peak.append(cur_sig[idx])
  plt.scatter(cur_x, cur_sig)
  plt.scatter(indexes, cur_peak, color='r')


  fig = plt.figure(3)
  indexes = peaks_finder_self(cur_sig)
  cur_peak = []
  for idx in indexes:
    cur_peak.append(cur_sig[idx])
  plt.scatter(cur_x, cur_sig)
  plt.scatter(indexes, cur_peak, color='r')  
  
  
  """
  indexes = find_peaks_cwt(cur_sig, np.arange(3, 6))
  indexes = index_revisor(indexes, cur_sig)
  """
  #indexes = peakutils.indexes(cur_sig, thres=1/max(cur_sig), min_dist=40)
    
  
    
  

  

  plt.show()






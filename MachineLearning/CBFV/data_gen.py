import datetime
import pickle
from random import shuffle, randint
import numpy as np

def floater(val):
  try:
    return float(val)
  except:
    return 0.0

def parameter_divider(params, para_name):
  lower_params = []
  for p in params:
    lower_params.append(p.lower())
  selected_parameter = []
  for i in range(0, len(lower_params)):
    if para_name in lower_params[i]:
      selected_parameter.append(params[i])
  return selected_parameter

class DataInfo:
  def __init__(self, line, params):
    self.patient = line[0]
    self.line_num = int(line[1])
    self.basic = {}
    self.spectral = {}
    self.HRV = {}
    self.BRS = {}
    self.morphology = {}
    self.phaseshift = {}
    self.others = {}
    self.fv_realated = {}
    self.fvx = 0
    for i in range(2, len(params)-1):
      analysis = params[i].split('-')[0]
      if analysis == 'BasicInfo':
        if 'FV' in params[i]:
          if 'last 10s mean' in params[i]:
            if line[i] != '' and line[i] != '0':
              self.fvx = floater(line[i])
          else:
            continue
        else:
          self.basic[params[i]]= floater(line[i])
      elif 'FV' in params[i].upper():
        self.fv_realated[params[i]] = floater(line[i])
      elif analysis == 'SpectralAnalysis':
        self.spectral[params[i]] = floater(line[i])
      elif analysis == 'MorphologyAnalysis':
        if 'Pulsatility' in params[i]:
          self.pi = floater(line[i])
        else:
          self.morphology[params[i]] = floater(line[i])
      elif analysis == 'HRV':
        self.HRV[params[i]] = floater(line[i])
      elif analysis == 'BRS':
        self.BRS[params[i]] = floater(line[i])
      elif analysis == 'PhaseShiftAnalysis':
        self.phaseshift[params[i]] = floater(line[i])
      else:
        if 'MX' in params[i]:
          self.mx = floater(line[i])
        else:
          self.others[params[i]] = floater(line[i])
    
class Other_info:
  def __init__(self, info):
    self.patient = info.patient
    self.line_num = info.line_num
    self.fvx = info.fvx
    self.prob = 0
    self.real = 0
    self.pred = 0
    self.corr = 0
  def infusion(self, val):
    self.prob = val[0]
    self.real = val[1]
    self.pred = val[2]
    self.corr = val[3]
    
def load_csv(path):
  info_list = []
  file = open(path, 'r')
  lines = file.readlines()
  file.close()
  params = lines[0].strip().split(',')
  for line in lines[1:]:
    split_line = line.split(',')
    if line_checker(split_line):
      cur_info = DataInfo(split_line, params)
      info_list.append(cur_info)
  return info_list

def gen_x(info_list, analysis = ['BAS', 'SPE', 'HRV', 'BRS', 'MOR', 'PHA', 'OTH'], diameter='abp'):
  X = []
  params = set([])
  other_list = []
  for dat in info_list:
    other_list.append(Other_info(dat))
    feature = []
    if 'BAS' in analysis:
      for k, v in dat.basic.items():
        if k in ['BasicInfo-ABP mean']:
          params.add(k)
          feature.append(v)
    if 'SPE' in analysis:
      spec_params = list(dat.spectral.keys())
      selected_parameter = parameter_divider(spec_params, diameter)
      for k, v in dat.spectral.items():
        if k in selected_parameter:
          params.add(k)
          feature.append(v)
    if 'HRV' in analysis:
      for k, v in dat.HRV.items():
        params.add(k)
        feature.append(v)
    if 'BRS' in analysis:
      for k, v in dat.BRS.items():
        params.add(k)
        feature.append(v)
    if 'MOR' in analysis:
      morp_params = list(dat.morphology.keys())
      selected_parameter = parameter_divider(morp_params, diameter)
      for k, v in dat.morphology.items():
        if k in selected_parameter:
          params.add(k)
          feature.append(v)
    if 'PHA' in analysis:
      for k, v in dat.phaseshift.items():
        params.add(k)
        feature.append(v)
    if 'OTH' in analysis:
      for k, v in dat.others.items():
        params.add(k)
        feature.append(v)
    X.append(feature)
  return X, list(params), other_list

def gen_x_using_params(info_list, abp_use=True, icp_use=True, fv_use=False, other_use=False, use_mode=[0, 1, 2]):
  abp_params=[]; icp_params=[]; fv_params=[]; other_params=[];
  if abp_use:
    file = open('rubbish/ABP_settings.csv', 'r')
    lines = file.readlines()
    for line in lines:
      split_line = line.split(',')
      if int(split_line[1]) in use_mode:
        abp_params.append(split_line[0])
    file.close()
  if icp_use:
    file = open('rubbish/ICP_settings.csv', 'r')
    lines = file.readlines()
    for line in lines:
      split_line = line.split(',')
      if int(split_line[1]) in use_mode:
        icp_params.append(split_line[0])
    file.close()
  if other_use:
    file = open('rubbish/Other_settings.csv', 'r')
    lines = file.readlines()
    for line in lines:
      split_line = line.split(',')
      if int(split_line[1]) in use_mode:
        other_params.append(split_line[0])
    file.close()
  if fv_use:
    file = open('rubbish/Fv_settings.csv', 'r')
    lines = file.readlines()
    for line in lines:
      split_line = line.split(',')
      if int(split_line[1]) in use_mode:
        fv_params.append(split_line[0])
    file.close()
  X = []
  for dat in info_list:
    feature = []
    for p in abp_params:
      feature.append(dat.abp_feature[p])
    for p in icp_params:
      feature.append(dat.icp_feature[p])
    for p in fv_params:
      feature.append(dat.fv_feature[p])
    for p in other_params:
      feature.append(dat.other_feature[p])
    X.append(feature)
  return X, abp_params+icp_params+fv_params+other_params

def gen_x_only_params(info_list, use_mode=[1]):
  file = open('rubbish/Dependency.csv' ,'r')
  lines = file.readlines()
  file.close()
  params = []
  for line in lines:
    split_line = line.split(',')
    if int(split_line[1]) in use_mode:
      params.append(split_line[0])
  X = []
  for dat in info_list:
    feature = []
    for p in params:
      if p in dat.abp_feature: feature.append(dat.abp_feature[p])
      if p in dat.icp_feature: feature.append(dat.icp_feature[p])
      if p in dat.other_feature: feature.append(dat.other_feature[p])
    X.append(feature)
  return X, params

def x_normalizer(X):
  trans_X = []
  mins = [10000 for i in range(0, len(X[0]))]
  maxs = [-10000 for i in range(0, len(X[0]))]
  for i in range(0, len(X)):
    for j in range(0, len(X[i])):
      if mins[j] > X[i][j]:
        mins[j] = X[i][j]
      if maxs[j] < X[i][j]:
        maxs[j] = X[i][j]
  for x in X:
    trans_line = []
    for i in range(0, len(x)):
      if maxs[i] - mins[i] == 0:
        trans_line.append(0)
      else:
        trans_line.append((x[i] - mins[i]) / (maxs[i] - mins[i]))
    trans_X.append(trans_line)
  return trans_X

def line_checker(line):
  zero_cnt = 0
  for v in line:
    if v == '0':
      zero_cnt += 1
  if zero_cnt > (len(line) * 0.9):
    return False
  else:
    return True

def gen_y_fvm(info_list, fvm_threshold):
  Y = []
  cnt = 0
  for dat in info_list:
    if dat.fvm < fvm_threshold:
      Y.append(0)
      cnt += 1
    else:
      Y.append(1)
  return Y, cnt

def gen_y_fvx(info_list, fvx_threshold):
  Y = []
  cnt = 0
  for dat in info_list:
    if dat.fvx < fvx_threshold:
      Y.append(0)
      cnt += 1
    else:
      Y.append(1)
  return Y, cnt

def gen_y_mx(info_list, mx_threshold):
  Y = []
  cnt = 0
  for dat in info_list:
    if dat.mx < mx_threshold:
      Y.append(0)
      cnt += 1
    else:
      Y.append(1)
  return Y, cnt

def gen_y_pi(info_list, pi_threshold):
  Y = []
  cnt = 0
  for dat in info_list:
    if dat.pi < pi_threshold:
      Y.append(0)
      cnt += 1
    else:
      Y.append(1)
  return Y, cnt

def gen_y_reduction(info_list, threshold):
  Y = []
  cnt = 0
  base_fvx = info_list[0].fvx
  for dat in info_list:
    reduce_ratio = (dat.fvx - base_fvx) / (base_fvx + 0.00001)
    if reduce_ratio < threshold:
      Y.append(0)
      cnt += 1
    else:
      Y.append(1)
  return Y, cnt



def gen_y_icp(info_list, icp_threshold):
  Y = []
  cnt = 0
  for dat in info_list:
    if dat.icp > icp_threshold:
      Y.append(0)
    else:
      Y.append(1)
  return Y, cnt

def pos_neg_regulator(X, Y, pos_cnt, neg_cnt):
  while(pos_cnt != neg_cnt):
    if pos_cnt > neg_cnt:
      idx = randint(0, len(Y) - 1)
      if Y[idx] == 1:
        X.append(X[idx])
        Y.append(1)
        neg_cnt += 1
    else:
      idx = randint(0, len(Y) - 1)
      if Y[idx] == 0:
        X.append(X[idx])
        Y.append(0)
        pos_cnt += 1
  """
  if pos_cnt > neg_cnt:
    for i in range(1, int(pos_cnt/neg_cnt)):
      for j in range(0, len(Y)):
        if Y[j] == 1:
          X.append(X[j])
          Y.append(1)
  else:
    for i in range(1, int(neg_cnt/pos_cnt)):
      for j in range(0, len(Y)):
        if Y[j] == 0:
          X.append(X[j])
          Y.append(0)
  """
  return X, Y

def gen_info_pickle():
  print('Start')
  cp1 = datetime.datetime.now()
  info_list = load_csv('new_sample_0703.csv')
  cp2 = datetime.datetime.now()
  print('Load finish ' + str(cp2 - cp1))
  filtered_info_list = []
  for info in info_list:
    if info.fvx > 5 and info.fvx < 300:
      filtered_info_list.append(info)
  print('Raw: ' + str(len(info_list)))
  print('Filtered: ' + str(len(filtered_info_list)))
  cp3 = datetime.datetime.now()
  print('Filtering finish ' + str(cp3 - cp2))
  return info_list

def shuffle_XY(X, Y):
  shf_X = []
  shf_Y = []
  index_shf = range(len(Y))
  shuffle(index_shf)
  for i in index_shf:
    shf_X.append(X[i])
    shf_Y.append(Y[i])
  return shf_X, shf_Y

def test_data_extractor(info_list, fold = 0):
  p_map = {}
  for p in info_list:
    k = p.patient[0:3]
    if k not in p_map:
      p_map[k] = 0
    p_map[k] += 1
  pat_list = list(p_map.keys())
  idxs = [i*5 + fold for i in range(0, int(len(pat_list)/5))]
  test_patient = []
  for i in idxs:
    test_patient.append(pat_list[i])
  print('Test patient: ' + str(len(test_patient)))
  print('Training patient: ' + str(len(pat_list) - len(test_patient)))
  test_info_list = []
  train_info_list = []
  for p in info_list:
    if p.patient[0:3] in test_patient:
      test_info_list.append(p)
    else:
      train_info_list.append(p)
  return train_info_list, test_info_list


    
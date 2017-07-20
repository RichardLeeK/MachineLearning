import numpy as np
import datetime
def parameter_load():
  file = open('rubbish/parameters.csv')
  lines = file.readlines()
  file.close()
  id = lines[0].split(',')[1]
  target= lines[1].split(',')[1]
  except_list = lines[2].strip().split(',')[1:]
  return id, target, except_list

def null_process(xs, mode='zero'):
  xs = np.array(xs)
  null_array = xs == -99999
  if mode == 'zero':
    xs[null_array] = 0
  elif mode == 'max':
    xs[null_array] = np.max(xs)
  elif mode == 'min':
    xs[null_array] = 99999
    xs[null_array] = np.min(xs)
  return xs

def norm(x):
  trans_x = []
  mins = [100000 for i in range(0, len(x[0]))]
  maxs = [-100000 for i in range(0, len(x[0]))]
  for i in range(0, len(x)):
    for j in range(0, len(x[i])):
      if mins[j] > x[i][j]:
        mins[j] = x[i][j]
      if maxs[j] < x[i][j]:
        maxs[j] = x[i][j]
  for X in x:
    trans_line = []
    for i in range(0, len(X)):
      if maxs[i] - mins[i] == 0:
        trans_line.append(0)
      else:
        trans_line.append((X[i] - mins[i])/(maxs[i] - mins[i]))
    trans_x.append(trans_line)
  return trans_x

def data_load(mode='train'):
  id, target, except_list = parameter_load()
  file = open('data/'+mode+'.csv')
  lines = file.readlines()
  file.close()
  header = lines[0].split(',')
  params_list = []; except_idx_list = []
  id_idx = 0; target_idx = 0;
  for i in range(len(header)):
    if header[i] == id: id_idx = i; continue;
    elif header[i] == target: target_idx = i; continue;
    elif header[i] in except_list:
      except_idx_list.append(i); continue;
    else:
      params_list.append(header[i])
  except_idx_list.append(id_idx)
  except_idx_list.append(target_idx)
  ids = []; xs = []; ys = [];
  for line in lines[1:]:
    sl = line.split(',')
    cur_id = sl[id_idx]
    cur_x = []
    cur_y = int(sl[target_idx])
    for i in range(len(sl)):
      if i not in except_idx_list:
        cur_x.append(float(sl[i]))
    ids.append(cur_id)
    xs.append(cur_x)
    ys.append(cur_y)
  xs = norm(null_process(xs))
  return ids, xs, ys, params_list

class SMInfo:
  # Shallow model information
  def __init__(self, id, *args):
    # id: LR, RF, GB, LS, KS
    self.id = id
    self.param = args

  def create_model(self):
    if self.id == 'LR':
      from sklearn.linear_model import LogisticRegression
      self.model = LogisticRegression()
    elif self.id == 'RF':
      from sklearn.ensemble import RandomForestClassifier
      self.model = RandomForestClassifier(n_estimators=self.param[0])
    elif self.id == 'GB':
      from sklearn.ensemble import GradientBoostingClassifier
      self.model = GradientBoostingClassifier(n_estimators=self.param[0])
    elif self.id == 'LS':
      from sklearn import svm
      self.model = svm.SVC(C=self.param[0],  kernel='linear')
    elif self.id == 'KS':
      from sklearn import svm
      self.model = svm.SVC(C=self.param[0], gamma=self.param[1], kernel='rbf')

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, y):
    pred = self.model.predict(x)
    tp = 0; tn = 0; fp = 0; fn = 0;
    for i in range(len(pred)):
      if y[i] == 0:
        if round(float(pred[i])) == 0: tp += 1
        else: fn += 1
      else:
        if round(float(pred[i])) == 0: fp += 1
        else: tn += 1
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    npd = (sen + spe) / 2
    acc = (tp + tn) / (tp + tn + fp + fn)
    return self.id+','+str(datetime.datetime.now())+','+str(tp)+','+str(tn)+','+str(fp)+','+str(fn)+','+str(sen)+','+str(spe)+','+str(ppv)+','+str(npv)+','+str(npd)+','+str(acc)+'\n'

def method_load():
  file = open('rubbish/methods.csv')
  lines = file.readlines()
  file.close()
  test_model = []
  for i in range(len(lines)):
    sl = lines[i].strip().split(',')
    if int(sl[1]) == 1:
      it = iter(sl[2:-1])
      params = []
      for v in it:
        if v == '': break
        if v == 'n_estimators':
          params.append(int(next(it)))
        else:
          params[v] = float(next(it))
      cur_model = SMInfo(sl[0], *params)
    else: continue
    test_model.append(cur_model)
  return test_model

if __name__=='__main__':
  print('Train data loading...')
  ids, xs, ys, params = data_load(mode='train')
  ids, vxs, vys, params = data_load(mode='test')
  print('Initial file loading...')
  test_model = method_load()
  for tm in test_model:
    print(tm.id + ' model is training...')
    tm.create_model()
    tm.train(xs, ys)
    print(tm.id + ' model is testing...')
    sentence = tm.predict(vxs, vys)
    pen = open('result.csv', 'a')
    pen.write(sentence)
    pen.close()
  print('All proccess are finished')
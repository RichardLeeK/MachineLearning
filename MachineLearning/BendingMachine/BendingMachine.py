import numpy as np
def parameter_load():
  file = open('rubbish/parameter.csv')
  lines = file.readlines()
  id = lines[0].split(',')[-1]
  target= lines[1].split(',')[-1]
  except_list = lines[2].split(',')[1:]
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

def norm(x, null_val):
  trans_x = []
  mins = [100000 for i in range(0, len(X[0]))]
  maxs = [-100000 for i in range(0, len(X[0]))]
  for i in range(0, len(x)):
    for j in range(0, len(x[i])):
      if mins[j] > x[i][j]:
        mins[j] = x[i][j]
      if maxs[j] < x[i][j]:
        maxs[j] = x[i][j]
  for X in x:
    trans_line = []
    for i in range(0, len(x)):
      if maxs[i] - mins[i] == 0:
        trans_line.append(0)
      else:
        trans_line.append((x[i] - mins[i])/(maxs[i] - mins[i]))
        trans_X.append(trans_line)
  return trans_x

def data_load():
  id, target, except_list = parameter_load()
  file = open('train/train.csv')
  lines = file.readlines()
  header = lines[0].split(',')
  params_list = []; except_idx_list = []
  for i in range(len(header)):
    if header[i] == id: id_idx = i; continue;
    elif header[i] == target: target_idx = i; continue;
    elif header[i] in except_list:
      except_idx_list.append(i); continue;
    else:
      params_list.append(head)
  except_idx_list.append(id_idx)
  except_idx_list.append(target_idx)
  ids = []; xs = []; ys = [];
  for line in lines[1:]:
    sl = line.split(',')
    cur_id = sl[id_idx]
    cur_x = []
    cur_y = sl[target_idx]
    for i in range(len(sl)):
      if i not in except_idx_list:
        cur_x.append(float(sl[i]))
    ids.append(cur_id)
    xs.append(cur_x)
    ys.append(cur_y)
  xs = norm(null_process(xs))
  return ids, xs, ys, params_list

def method_load():


if __name__=='__main__':
  ids, xs, ys, params = data_load()

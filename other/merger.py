import os

class FileInfo:
  def __init__(self, lines):
    self.features = lines[0].strip().split(',')[1:]
    self.data = {}
    self.file = lines[1].split(',')[0].split('_')[0]
    for line in lines[1:]:
      sl = line.split(',')
      v = {}
      for i in range(1, len(sl)):
        if 'NULL' in sl[i]:
          v[self.features[i - 1]] = 0
        else:
          v[self.features[i - 1]] = float(sl[i])
      self.data[int(sl[0].split('_')[1])] = v

path = 'D:/Richard/CBFV/MMA result/'
folders = os.listdir(path)
data_list = []
feature_set = set([])
for fold in folders:
  if fold == 'no': continue
  if len(fold.split('.')) < 2:
    print(fold)
    file = open(path + fold + '/ResultFile.csv')
    lines = file.readlines()
    file_division = {}
    cur_file_division = [lines[0]]
    cur_file_name = lines[1].split(',')[0].split('_')[0]
    for line in lines[1:]:
      fileName = line.split(',')[0].split('_')[0]
      ln = int(line.split(',')[0].split('_')[-1])
      if line.split(',')[0].split('_')[0] not in file_division:
        file_division[fileName] = {0: lines[0]}
      file_division[fileName][ln] = line
    for fdk, fdv in file_division.items():
      cur_lines = []
      for k, v in fdv.items():
        cur_lines.append(v)
      data_list += [FileInfo(cur_lines)]
    header = lines[0].split(',')
    for h in header:
      feature_set.add(h.strip())
fs = list(feature_set)
pen = open('new_sample_0703.csv', 'w')
sentence = 'File,LN'
for v in fs:
  sentence += (',' + v)
pen.write(sentence + '\n')
for dl in data_list:
  for k, v in dl.data.items():
    sentence = dl.file + ',' + str(k)
    for f in fs:
      if f in dl.features:
        sentence += ',' + str(v[f])
      else:
        sentence += ','
    pen.write(sentence + '\n')
pen.close()
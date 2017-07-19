import os
from datetime import datetime, timedelta
import pytz
from statistics import mean
 
def fromOADate(v):
  return datetime(1899, 12, 30, 0, 0, 0, tzinfo=pytz.utc) + timedelta(days=v)

def find_bound(source_dic, time):
  key_list = list(source_dic.keys())
  key_list.sort()
  start = 0
  end = len(key_list)
  while(True):
    idx = int((start+end)/2)
    if time < key_list[idx]:
      end = idx
    else:
      start = idx
    if start == end or start+1 == end:
      break;
  return key_list[start]


def map_squeeze(data_map):
  sq_map = {}
  for k, v in data_map.items():
    if len(data_map[k]) == 0:
      sq_map[k] = 0
    else:
      sq_map[k] = mean(data_map[k])
  return sq_map

def file_analysis(path, path2, file):
  f = open(path + '/' + file)
  lines = f.readlines()
  f.close()
  parameter = lines[0].split(',')[1:]
  super_map = {}
  data_map = {}
  for p in parameter:
    data_map[p] = []
  bef_time = fromOADate(float(lines[1].split(',')[0]))
  for line in lines[1:]:
    split_line = line.split(',')
    cur_time = fromOADate(float(split_line[0]))
    for i in range(0, len(parameter)):
      if split_line[i+1] == '': continue
      if split_line[i+1] == '\n': continue
      if split_line[i+1] == 'nan': continue
      data_map[parameter[i]].append(float(split_line[i+1]))
    if (cur_time - bef_time).total_seconds() >= 5:
      super_map[bef_time] = map_squeeze(data_map)
      for p in parameter:
        data_map[p] = []
      bef_time = cur_time

  f2 = open(path2 + '/' + file.split('_')[0] + '.csv', 'r')
  raw_lines = f2.readlines()
  f2.close()
  raw_map = {}
  raw_data_map = {}
  bef_time = fromOADate(float(lines[1].split(',')[0]))
  raw_parameter = raw_lines[0].split(',')[1:]
  for p in raw_parameter:
    raw_data_map[p.strip()] = []
  for line in raw_lines[1:]:
    split_line = line.split(',')
    cur_time = fromOADate(float(split_line[0]))
    for i in range(0, len(raw_parameter)):
      if split_line[i+1] == '\n': continue
      if split_line[i+1] == '': continue
      if float(split_line[i+1]) < 0: continue
      raw_data_map[raw_parameter[i].strip()].append(float(split_line[i+1]))
    if (cur_time - bef_time).total_seconds() >= 5:
      raw_map[bef_time] = map_squeeze(raw_data_map)
      for p in raw_parameter:
        raw_data_map[p] = []
      bef_time = cur_time

  return super_map, raw_map


if __name__ ==  '__main__':
  path = 'D:/YT/Result/FV decrease study merge whole'
  path2 = 'D:/YT/Data/820 TBI csv files (FV)'
  params = open('params.csv').readline()
  param = params.split(',')[2:]
  raw_param = ['abp[abp]', 'icp[icp]', 'fvm[fvm]', 'fvx[fvx]', 'fvr[fvr]', 'fvl[fvl]', 'flu[flu]']
  files = os.listdir(path)
  raw_files = os.listdir(path2)
  pen = open('AllBatchMergedFile.csv', 'w')
  pen.write(params.strip() + ',abp[abp],icp[icp],fvm[fvm],fvx[fvx],fvr[fvr],fvl[fvl],flu[flu]' + '\n')
  for file in files:
    print(file)
    super_map, raw_map = file_analysis(path, path2, file)
    for k, v in super_map.items():
      sen = file.split('_')[0] + ',' + k.strftime('%Y-%m-%d %H:%M:%S')
      for p in param:
        if p in v:
          sen += (',' + str(v[p]))
        else:
          sen += ',0'
      for rp in raw_param:
        raw_idx = find_bound(raw_map, k)
        if rp in raw_map[raw_idx]:
          sen += (',' + str(raw_map[raw_idx][rp]))
        else:
          sen += ',0'
      pen.write(sen+'\n')
  pen.close()

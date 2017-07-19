import os
import numpy as np
import threading
import shutil

def folder_process(path, path2, folder):
  parameters = []
  times = []
  files = os.listdir(path + folder)
  for file in files:
    try: 
      if len(file.split('.')) < 2: continue
      if file == 'BasicInfo.csv': continue
      if file == 'Merged.csv': continue
      f = open(path + folder + '/' + file, 'r')
      lines = f.readlines()
      cor_name = lines[0].split(',')
      for v in cor_name:
        parameters.append(v.strip())
      for line in lines[1:]:
        time = float(line.split(',')[0])
        times.append(time)
      f.close()
    except:
      pen = open(path2 + 'exception.txt', 'a')
      pen.write(folder + '\t' + file + '\n')
      pen.close()
  parameters = list(set(parameters))
  times = list(set(times))
  parameters.sort()
  times.sort()
  data_array = np.zeros((len(times), len(parameters)))
    
  print(folder + ' start')
  for file in files:
    try:
      if len(file.split('.')) < 2: continue
      if file == 'BasicInfo.csv': continue
      if file == 'Merged.csv': continue
      f = open(path + folder + '/' + file, 'r')
      lines = f.readlines()
      cur_params = lines[0].split(',')
      for line in lines[1:]:
        split_line = line.split(',')
        for i in range(1, len(cur_params)):
          if split_line[i] == ' ': continue
          if split_line[i] == ' \n': continue
          if split_line[i] == '': continue
          if split_line[i] == '\n': continue
          try:
            parm_index = parameters.index(cur_params[i].strip())
            time_index = times.index(float(split_line[0].strip()))
            data_array[time_index][parm_index] = float(split_line[i].strip())
          except:
            print(split_line[i] + ' can`t be transformed')
    except:
      pen = open(path2 + 'exception.txt', 'a')
      pen.write(folder + '\t' + file + '\n')
      pen.close()

  pen = open(path + folder + '/Merged.csv', 'w')
  sentence = ''
  for v in parameters:
    sentence += (v + ',')
  pen.write(sentence + '\n')
  i = 0
  for values in data_array:
    sentence = str(times[i])
    for v in values:
      if v == 0:
        sentence += ','
      else:
        sentence += (str(v) + ',')
    pen.write(sentence + '\n')
    i += 1
  pen.close()
  shutil.copy(path + folder + '/Merged.csv', path2 + folder+'_merged.csv')

  print(folder + ' finished')


if __name__ ==  '__main__':
  
  path = 'D:/YT/Result/FV decrease baro_spectral_hrv_morp/Summary/'
  path2 = 'D:/YT/Result/FV decrease baro_spectral_hrv_morp/'
  folders = os.listdir(path)
  ths = []
  process_cnt = 0
  for folder in folders:
    if os.path.exists(path2 + folder + '_merged.csv'): 
      process_cnt += 1
      continue
    th = threading.Thread(target=folder_process, args=(path, path2, folder))
    th.start()
    ths.append(th)
    if len(ths) > 200:
      for th in ths:
        th.join()
      process_cnt += len(ths)
      print(str(process_cnt) + ' / ' + str(len(folders)) + ' processed')
      ths = []
  for th in ths:
    th.join()
  print('all process is finished')
  """
  folder_process('D:/YT/Result/FV Decrease HRV and baroreflex/Summary/', 'D:/YT/Result/FV Decrease HRV and baroreflex/', '023110GS')
  """
    





      

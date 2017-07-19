from datetime import datetime, timedelta
import pytz

def fromOADate(v):
  return datetime(1899, 12, 30, 0, 0, 0, tzinfo=pytz.utc) + timedelta(days=v)

ori = open('rev_merged_dat2_test.csv')
lines = ori.readlines()
ori.close()

def fvx_idx(header):
  for i in range(0, len(header)):
    if 'mx' in header[i].lower():
      return i
  return -1

def get_fvx(line, fvxIdx):
  sl = line.split(',')
  if fvxIdx < 0:
    return (float(sl[-1]) + float(sl[-2]))/2
  else:
    return float(sl[fvxIdx])

ori_map = {}
for line in lines[1:]:
  sl = line.split(',')
  if sl[0] not in ori_map:
    ori_map[sl[0]] = {}
  ori_map[sl[0]][int(sl[1])] = line.strip()

patient_path = 'D:/820 TBI csv files (FV)/'
for k, v in ori_map.items():
  cur_file = open(patient_path+str(k)+'.csv')
  cur_lines = cur_file.readlines()
  fvxIdx = fvx_idx(cur_lines[0].split(','))
  sta_time = fromOADate(float(cur_lines[1].split(',')[0]))
  for i in range(2, len(cur_lines)):
    sl = cur_lines[i].split(',')
    cur_time = fromOADate(float(sl[0]))
    if (cur_time - sta_time).total_seconds() >= 300:
      idx300 = i
      break
  bef_time = fromOADate(float(cur_lines[idx300].split(',')[0]))
  fvx_buff = []; m_fvx_buff = []
  for i in range(idx300, len(cur_lines)):
    sl = cur_lines[i].split(',')
    cur_time = fromOADate(float(sl[0]))
    fvx_buff.append(get_fvx(cur_lines[i], fvxIdx))
    if (cur_time - bef_time).total_seconds() >= 10:
      m_fvx_buff.append(sum(fvx_buff)/len(fvx_buff))
      fvx_buff = []
      bef_time = cur_time
  print('abc')



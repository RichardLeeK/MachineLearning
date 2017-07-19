
import sys
sys.path.insert(0, 'D:/Sources/Python Source Code')
import img_process.signal_to_img as si

file = open('D:/Richard/CBFV/Auto-encoder/001040SE_interpolated.csv')

lines = file.readlines()
file.close()

singal_map = {}

for line in lines:
  sl = line.split(',')
  cur_sig = []
  for v in sl[1:]:
    cur_sig.append(float(v))
  sl[int(line[0])] = cur_sig

val = si.interpolated_signal_to_img(sl[1])

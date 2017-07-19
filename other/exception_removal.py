exc = open('rubbish/exception.csv')
exc_file = exc.readlines()
exc.close()
data = open('merged_data3.csv')
data_line = data.readlines()
data.close()
pen = open('rev_merged_dat3.csv', 'w')
pen.write(data_line[0].strip() + ',FVX_L\n')
cnt = 0

header = data_line[0].split(',')
rl_idx = header.index('BasicInfo-FVR+L mean')
x_idx = header.index('BasicInfo-FVX mean')
for dl in data_line[1:]:
  if (dl.split(',')[0] + '\n') not in exc_file:
    if dl.split(',')[rl_idx] == '' and dl.split(',')[x_idx] == '':
      print('PROB')
    elif dl.split(',')[rl_idx] == '' or dl.split(',')[rl_idx] == '0':
      pen.write(dl.strip()+','+dl.split(',')[x_idx]+'\n')
    else:
      pen.write(dl.strip()+','+dl.split(',')[rl_idx]+'\n')
  cnt += 1
  print(str(cnt / len(data_line)))
pen.close()
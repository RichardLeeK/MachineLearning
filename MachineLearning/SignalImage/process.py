def gos_loader():
  file = open('rubbish/GOS.csv')
  lines = file.readlines()
  dic = {}
  for line in lines:
    sl = line.split(',')
    dic[sl[0]] = sl[1]

  return dic

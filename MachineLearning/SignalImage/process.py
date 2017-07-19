def gos_loader():
  file = open('rubbish/GOS.csv')
  lines = file.readlines()
  gos_map = {}
  for line in lines[1:]:
    sl = line.split(',')
    gos_map[int(sl[0])] = int(sl[1])
    return gos_map

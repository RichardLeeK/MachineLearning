file = open('reject.csv')
lines = file.readlines()
ori = []
rej = []
for line in lines:
  sl = line.split(',')
  ori.append(sl[1].strip())
  rej.append(sl[0].strip())

pen = open('remain.csv' , 'w')
for o in ori:
  if o not in rej:
    pen.write(o + '\n')
pen.close()
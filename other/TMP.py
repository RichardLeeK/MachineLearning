file = open('new_sample_0703.csv')

lines = file.readlines()
file.close()

pen = open('features.csv', 'w')

sl = lines[0].split(',')
for s in sl:
  if 'SpectralAnalysis' in s:
    if 'ABP' in s:
      pen.write('Spectral,ABP,'+s+'\n')
    elif 'ICP' in s:
      pen.write('Spectral,ICP,'+s+'\n')
    else:
      pen.write('Spectral,OTH,'+s+'\n')
  elif 'HRV' in s:
    pen.write('HRV,,'+s+'\n')
  elif 'MorphologyAnalysis' in s:
    if 'Abp' in s:
      pen.write('Spectral,ABP,'+s+'\n')
    elif 'Icp' in s:
      pen.write('Spectral,ICP,'+s+'\n')
    else:
      pen.write('Spectral,OTH,'+s+'\n')
  elif 'BasicInfo' in s:
    pen.write('Basic,,'+s+'\n')
  elif 'BRS' in s:
    pen.write('BRS,,'+s+'\n')
pen.close()
print('fin')


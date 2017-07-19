import sys
sys.path.insert(0, 'D:/Sources/Python Source Code')
import data_gen as dg

def patient_distribution(info_list, threshold):
  pdm = {}
  for v in info_list:
    if v.patient[:3] not in pdm:
      pdm[v.patient[:3]] = [0, 0]
    if v.fvx < threshold:
      pdm[v.patient[:3]][0] += 1
    else:
      pdm[v.patient[:3]][1] += 1
  return pdm

threshold = 55
info_list = dg.gen_info_pickle()
print('Load pickle finish')
pdm = patient_distribution(info_list, threshold)
pen = open('Data distribution_' + str(threshold) + '.csv', 'w')
pen.write('Patient,Positive,Negative,PN,NN\n')
for k, v in pdm.items():
  sum = v[0] + v[1]
  pen.write(str(k) + ',' + str(v[0]) + ',' + str(v[1]) + ',' + str(v[0]/sum) + ',' + str(v[1]/sum) + '\n')
print('Finish')

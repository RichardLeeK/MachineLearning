import numpy as np
import sys
sys.path.insert(0, 'D:/Sources/Python Source Code')
import data_gen as dg

# Pearson correlation coefficient
def correlation(X, Y):
  return np.corrcoef(X, Y)[1, 0]

# Mutual Information
def mutual_information(x, y):
  sum_mi = 0.0
  x_value_list = np.unique(x)
  y_value_list = np.unique(y)
  Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
  Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
  for i in range(0, len(x_value_list)):
    if Px[i] ==0.:
      continue
    sy = y[x == x_value_list[i]]
    if len(sy)== 0:
      continue
    pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
    t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
    sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
  return sum_mi

def fs_using_CC():
  file = open('CCMap.csv', 'r')
  
def rev_dep_using_CC_map():
  file = open('CCMap.csv', 'r')
  lines = file.readlines()
  fvx_map = {}
  orj_sen = 'origin'
  rej_sen = 'reject'

  for line in lines:
    sl = line.split(',')
    if sl[0] == 'FVX':
      fvx_map[sl[1]] = float(sl[2].strip())
      orj_sen += ',' + sl[1]
    else:
      if float(sl[2].strip()) > 0.5 or float(sl[2].strip()) < -0.5:
        first_corr = fvx_map[sl[0]]
        second_corr = fvx_map[sl[1]]
        if first_corr >= second_corr:
          rej_sen += ',' + sl[1]
        else:
          rej_sen += ',' + sl[0]
  pen = open('reject.csv', 'w')
  pen.close()



def gen_CC_map():
  info_list = dg.gen_info_pickle()
  X, params = dg.gen_x_using_params(info_list, fv_use=True, other_use=True, use_mode=[0, 1, 2])
  fvxs = []
  for info in info_list:
    fvxs.append(info.fvx)
  npX = np.array(X)
  npX = npX.transpose()
  pen = open('CCMap.csv', 'w')
  for i in range(0, len(npX)):
    pen.write('FVX,'+params[i]+','+str(correlation(fvxs, npX[i]))+'\n')
  for i in range(0, len(npX)):
    for j in range(0, len(npX)):
      if i == j: continue
      print(params[i] + '\t' + params[j] + ' : ' + str(correlation(npX[i], npX[j])))
      pen.write(params[i] + ',' + params[j] + ',' + str(correlation(npX[i], npX[j])) + '\n')
  pen.close()

if __name__ == "__main__":
  gen_CC_map()
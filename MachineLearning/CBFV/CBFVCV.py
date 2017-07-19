import pickle
import datetime
import ml.shallow_model as sm
import ml.deep_model as dl
import data_gen as dg
import sys
import numpy as np
import threading

parameter_class = 'HRV,SPE,MOR(3)R'
threshold = 1.3
fold = 5

mode = parameter_class + '_' + str(threshold) + '_' + str(fold)

def pos_counter(Y):
  cnt = 0
  for y in Y:
    if y == 0:
      cnt += 1
  return cnt, len(Y) - cnt

if __name__ == '__main__':
  sys.setrecursionlimit(10000)
  print(mode)
  info_list = dg.gen_info_pickle()
  cv_list = dg.cross_validation_DS_gen(info_list, fold)
  for i in range(fold):
    train_list = []
    for j in range(fold):
      if i == j: continue
      train_list += cv_list[j]
    valid_list = cv_list[i]
    X_train, param, _ = dg.gen_x(train_list, ['BAS', 'HRV', 'SPE', 'MOR'], '')
    X_test, _, _ = dg.gen_x(valid_list, ['BAS', 'HRV', 'SPE', 'MOR'], '')
    print('Use ' + str(len(param)) + ' features!')
    pen = open('rubbish/params.csv', 'a')
    sentence = '\n' + mode
    for p in param:
      sentence += ',' + p
    pen.write(sentence)
    pen.close()

    Y_train, cnt = dg.gen_y_pi(train_list, threshold)
    Y_test, cnt_test = dg.gen_y_pi(valid_list, threshold)

    X_train, Y_train = dg.pos_neg_regulator(X_train, Y_train, cnt, len(Y_train) - cnt)

    
    p, n = pos_counter(Y_train)
    pp, nn = pos_counter(Y_test)
    print('Y_train: ' + str(p) + '\t' + str(n))
    print('Y_test: ' + str(pp) + '\t' + str(nn))
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    line = mode + ',' + str(i) +',' + sm.rf_train_test(X_train, X_test, Y_train, Y_test)
    pen = open('CVRes.csv', 'a')
    pen.write(line + '\n')
    pen.close()

    line = mode + ',' + str(i) + ',' + sm.lr_train_test(X_train, X_test, Y_train, Y_test)
    pen = open('CVRes.csv', 'a')
    pen.write(line + '\n')
    pen.close()

    line = mode + ',' + str(i) + ',' + sm.gb_train_test(X_train, X_test, Y_train, Y_test)
    pen = open('CVRes.csv', 'a')
    pen.write(line + '\n')
    pen.close()

    line = mode + ',' + str(i) + ',' + dl.dnn_train_test(X_train, X_test, Y_train, Y_test)
    pen = open('CVRes.csv', 'a')
    pen.write(line + '\n')
    pen.close()

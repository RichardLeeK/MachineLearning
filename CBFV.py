from sklearn import svm
import pickle
import datetime
import sklearn.cross_validation as cv
import ml.shallow_models as sm
import ml.deep_models as dl
import data_gen as dg
import sys
import numpy as np
import copy
import threading

test_param = 'PI'
parameter_class = 'REG_HRV+SPE(ABP)'
fvx_threshold = 1.56
fold = 4
mode = test_param + '_' + parameter_class + '_' + str(fvx_threshold)+ '_' + str(fold) 

def pos_counter(Y):
  cnt = 0
  for y in Y:
    if y == 0:
      cnt += 1
  return cnt, len(Y) - cnt

if __name__ == "__main__":
  sys.setrecursionlimit(10000)
  print(mode)
  info_list = dg.gen_info_pickle()
  info_list, test_info_list = dg.test_data_extractor(info_list, fold)
  cp3 = datetime.datetime.now()
  X_train, params, _ = dg.gen_x(info_list, ['BAS', 'HRV', 'SPE'], 'abp')
  X_test, _, OL_test = dg.gen_x(test_info_list, ['BAS', 'HRV', 'SPE'], 'abp')

  print('Use ' + str(len(params)) + ' features!')
  pen = open('rubbish/params.csv', 'a')
  sentence = '\n' + mode
  for p in params:
    sentence += ',' + p
  pen.write(sentence)
  pen.close()
  
  Y_train, cnt = dg.gen_y_pi(info_list, fvx_threshold)
  Y_test, cnt_test = dg.gen_y_pi(test_info_list, fvx_threshold)


  """
  Y_train, cnt = dg.gen_y_fvx(info_list, fvx_threshold)
  Y_test, cnt_test = dg.gen_y_fvx(test_info_list, fvx_threshold)
  
  Y_train, cnt = dg.gen_y_reduction(info_list, 0.2)
  Y_test, cnt_test = dg.gen_y_reduction(test_info_list, 0.2)
  """
  cp4 = datetime.datetime.now()
  print('Generate data fininshed ' + str(cp4 - cp3))
  print('Train Positive: ' + str(cnt))
  print('Train Negative: ' + str(len(Y_train)-cnt))
  print('Test Positive: ' + str(cnt_test))
  print('Test Negative: ' + str(len(Y_test)-cnt_test))
  """ Regulization
  X, Y = dg.pos_neg_regulator(X, Y, cnt, len(Y) - cnt)
  pos_cnt, neg_cnt = pos_counter(Y)
  print('Regulated Positive: ' + str(pos_cnt))
  print('Regulated Negative: ' + str(neg_cnt))
  """
  X_train, Y_train = dg.pos_neg_regulator(X_train, Y_train, cnt, len(Y_train) - cnt)
  pos_cnt, neg_cnt = pos_counter(Y_train)
  print('Regulated Positive: ' + str(pos_cnt))
  print('Regulated Negative: ' + str(neg_cnt))

  #X_train, Y_train = dg.pos_neg_regulator(X_train, Y_train, cnt, len(Y_train)-cnt)
  #pos_cnt, neg_cnt = pos_counter(Y_train)
  #print('Regulated Train Positive: ' + str(pos_cnt))
  #print('Regulated Train Negative: ' + str(neg_cnt))

  X_train = dg.x_normalizer(X_train)
  X_test = dg.x_normalizer(X_test)

  p, n = pos_counter(Y_train)
  pp, nn = pos_counter(Y_test)
  print('Y_train: ' + str(p) + '\t' + str(n))
  print('Y_test: ' + str(pp) + '\t' + str(nn))

  X_train = np.array(X_train)
  X_test = np.array(X_test)
  Y_train = np.array(Y_train)
  Y_test = np.array(Y_test) 
  
  with open('data/dataset.pickle', 'wb') as handle:
    pickle.dump({'X_train': X_train, 'y_train': Y_train, 'X_test': X_test, 'y_test': Y_test}, handle, protocol=pickle.HIGHEST_PROTOCOL)

  print('All process is finished')

  """
  rf_OL_test = copy.deepcopy(OL_test)
  line = mode + ',' + sm.rf_train_test(X_train, X_test, Y_train, Y_test, rf_OL_test)
  rf_observe = open(str(fold) + '_rf_observ.csv', 'w')
  rf_observe.write('Case,LN,fvx,prob,real,pred,corr\n')
  for ot in rf_OL_test:
    rf_observe.write(ot.patient + ',' + str(ot.line_num) + ',' + str(ot.fvx) + ',' + str(ot.prob) + ',' + str(ot.real) + ',' + str(ot.pred) + ',' + str(ot.corr) + '\n')
  rf_observe.close()
  pen = open('MLResult.csv', 'a')
  pen.write(line + '\n')
  pen.close()

  lr_OL_test = copy.deepcopy(OL_test)
  line = mode + ',' + sm.lr_train_test(X_train, X_test, Y_train, Y_test, lr_OL_test)
  pen = open('MLResult.csv', 'a')
  pen.write(line + '\n')
  pen.close()

  gb_OL_test = copy.deepcopy(OL_test)
  line = mode + ',' + sm.gb_train_test(X_train, X_test, Y_train, Y_test, gb_OL_test)
  pen = open('MLResult.csv', 'a')
  pen.write(line + '\n')
  pen.close()

  line = mode + ',' + dl.dnn_train_test(X_train, X_test, Y_train, Y_test)
  pen = open('MLResult.csv', 'a')
  pen.write(line + '\n')
  pen.close()
  
  line = mode + ',' + dl.lstm_train_test(X_train, X_test, Y_train, Y_test)
  pen = open('MLResult.csv', 'a')
  pen.write(line + '\n')
  pen.close()
  
  line = mode + ',' + sm.svm_train_test(X_train, X_test, Y_train, Y_test)
  pen = open('MLResult.csv', 'a')
  pen.write(line + '\n')
  pen.close()
  """
from sklearn import svm
import pickle
import datetime
import sklearn.cross_validation as cv
import ml.shallow_models as sm
import ml.deep_models as dl
import data_gen as dg
import sys
import numpy as np

mode = 'ABP_SEP_FORICP'

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
  info_list, test_info_list = dg.test_data_extractor(info_list)
  cp3 = datetime.datetime.now()
  #X_train, params = dg.gen_x(info_list, is_abp_use=False, is_icp_use=False, is_other_use=True)
  #X_test, _ = dg.gen_x(test_info_list, is_abp_use=False, is_icp_use=False, is_other_use=True)
  X_train, params = dg.gen_x_using_params(info_list, abp_use=True, icp_use=False, other_use=False, use_mode=[0, 1])
  X_test, _ = dg.gen_x_using_params(test_info_list, abp_use=True, icp_use=False, other_use=False, use_mode=[0, 1])
  print('Use ' + str(len(params)) + ' features!')
  pen = open('rubbish/params.csv', 'a')
  sentence = '\n' + mode
  for p in params:
    sentence += ',' + p
  pen.write(sentence)
  pen.close()
  Y_train, cnt = dg.gen_y_icp(info_list, 22)
  Y_test, cnt_test = dg.gen_y_icp(test_info_list, 22) 
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

  with open('data/dataset.pickle', 'wb') as handle:
    pickle.dump({'X_train':X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test}, handle, protocol=pickle.HIGHEST_PROTOCOL)

  X_train = np.array(X_train)
  X_test = np.array(X_test)
  Y_train = np.array(Y_train)
  Y_test = np.array(Y_test)
  line = mode + ',' + sm.rf_train_test(X_train, X_test, Y_train, Y_test)
  pen = open('MLResult.csv', 'a')
  pen.write(line + '\n')
  pen.close()
  line = mode + ',' + sm.lr_train_test(X_train, X_test, Y_train, Y_test)
  pen = open('MLResult.csv', 'a')
  pen.write(line + '\n')
  pen.close()
  line = mode + ',' + sm.svm_train_test(X_train, X_test, Y_train, Y_test)
  pen = open('MLResult.csv', 'a')
  pen.write(line + '\n')
  pen.close()
  line = mode + ',' + sm.gb_train_test(X_train, X_test, Y_train, Y_test)
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
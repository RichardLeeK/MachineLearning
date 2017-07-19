from ml.public import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
#import ml.sequential_selection as ss
import pickle
import datetime

sbs_num = 50
 
# Support vector machine
def svm_train_test(X_train, X_test, Y_train, Y_test):
  st = datetime.datetime.now()
  clf = svm.SVC()
  """
  sbs = ss.SBS(clf, sbs_num)
  sbs.fit(X_train, Y_train)
  X_train = sbs.transform(X_train)
  X_test = sbs.transform(X_test)
  """
  clf.fit(X_train, Y_train)
  with open('network/svm_s.net', 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
  Y_pred = clf.predict(X_test)
  tp, tn, fp, fn = pred_test(Y_pred, Y_test)
  ed = datetime.datetime.now()
  print('Support Vector Machine finished ' + str(ed - st))
  print('TP: ' + str(tp) + '\tTN: ' + str(tn) + '\tFP: ' + str(fp) + '\tFN: ' + str(fn))
  return 'SVM,' + gen_result_line(tp, tn, fp, fn)

# Random forest
def rf_train_test(X_train, X_test, Y_train, Y_test, OL_test):
  st = datetime.datetime.now()
  rf = RandomForestClassifier(n_estimators=500)
  """
  sbs = ss.SBS(rf, sbs_num)
  sbs.fit(X_train, Y_train)
  X_train = sbs.transform(X_train)
  X_test = sbs.transform(X_test)
  """
  rf.fit(X_train, Y_train)
  with open('network/rf_s.net', 'wb') as handle:
    pickle.dump(rf, handle, protocol=pickle.HIGHEST_PROTOCOL)
  Y_pred = rf.predict(X_test)
  Y_prob = rf.predict_proba(X_test)
  for i in range(0, len(Y_pred)):
    OL_test[i].infusion([Y_prob[i][-1], Y_test[i], Y_pred[i], 1 if Y_pred[i] == Y_test[i] else 0])
  tp, tn, fp, fn = pred_test(Y_pred, Y_test)
  ed = datetime.datetime.now()
  print('Random Forest fininshed ' + str(ed - st))
  print('TP: ' + str(tp) + '\tTN: ' + str(tn) + '\tFP: ' + str(fp) + '\tFN: ' + str(fn))
  return 'Random Forest,' + gen_result_line(tp, tn, fp, fn)

def gb_train_test(X_train, X_test, Y_train, Y_test, OL_test):
  st = datetime.datetime.now()
  gb = GradientBoostingClassifier(n_estimators=500)
  """
  sbs = ss.SBS(gb, sbs_num)
  sbs.fit(X_train, Y_train)
  X_train = sbs.transform(X_train)
  X_test = sbs.transform(X_test)
  """
  gb.fit(X_train, Y_train)
  with open('network/gb_s.net', 'wb') as handle:
    pickle.dump(gb, handle, protocol=pickle.HIGHEST_PROTOCOL)
  Y_pred = gb.predict(X_test)
  Y_prob = gb.predict_proba(X_test)
  for i in range(0, len(Y_pred)):
    OL_test[i].infusion([Y_prob[i][-1], Y_test[i], Y_pred[i], 1 if Y_pred[i] == Y_test[i] else 0])
  tp, tn, fp, fn = pred_test(Y_pred, Y_test)
  ed = datetime.datetime.now()
  print('Gradient Boosting finished ' + str(ed - st))
  print('TP: ' + str(tp) + '\tTN: ' + str(tn) + '\tFP: ' + str(fp) + '\tFN: ' + str(fn))
  return 'Gradient Boosting,' + gen_result_line(tp, tn, fp, fn)

def lr_train_test(X_train, X_test, Y_train, Y_test, OL_test):
  st = datetime.datetime.now()
  lr = LogisticRegression()
  """
  sbs = ss.SBS(lr, sbs_num)
  sbs.fit(X_train, Y_train)
  X_train = sbs.transform(X_train)
  X_test = sbs.transform(X_test)
  """
  lr.fit(X_train, Y_train)
  with open('network/lr_s.net', 'wb') as handle:
    pickle.dump(lr, handle, protocol=pickle.HIGHEST_PROTOCOL)
  Y_pred = lr.predict(X_test)
  Y_prob = lr.predict_proba(X_test)
  for i in range(0, len(Y_pred)):
    OL_test[i].infusion([Y_prob[i][-1], Y_test[i], Y_pred[i], 1 if Y_pred[i] == Y_test[i] else 0])
  tp, tn, fp, fn = pred_test(Y_pred, Y_test)
  ed = datetime.datetime.now()
  print('Logistic Regression finished ' + str(ed - st))
  print('TP: ' + str(tp) + '\tTN: ' + str(tn) + '\tFP: ' + str(fp) + '\tFN: ' + str(fn))
  return 'Logistic Regression,' + gen_result_line(tp, tn, fp, fn)
def pred_test(Y_pred, Y_test):
  tp = 0; tn = 0; fp = 0; fn = 0
  for i in range(0, len(Y_pred)):
    if Y_test[i] == 0:
      if round(Y_pred[i]) == 0:
        tp += 1
      else:
        fn += 1
    else:
      if round(Y_pred[i]) == 0:
        fp += 1
      else:
        tn += 1
  return tp, tn, fp, fn

def pred_test_lstm(Y_pred, Y_test):
  tp = 0; tn = 0; fp = 0; fn = 0
  for i in range(0, len(Y_pred)):
    if Y_test[i] == 0:
      if round(Y_pred[i][0]) == 0:
        tp += 1
      else:
        fn += 1
    else:
      if round(Y_pred[i][0]) == 0:
        fp += 1
      else:
        tn += 1
  return tp, tn, fp, fn

def gen_result_line(tp, tn, fp, fn):
  sen = tp / (tp + fn) if (tp + fn) > 0 else 0
  spe = tn / (tn + fp) if (tn + fp) > 0 else 0
  ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
  npv = tn / (tn + fn) if (tn + fn) > 0 else 0
  npd = (sen + spe) / 2
  acc = (tp + tn) / (tp + tn + fp + fn)
  line = str(tp) + ',' + str(tn) + ',' + str(fp) + ',' + str(fn)
  line += ',' + str(sen) + ',' + str(spe) + ',' + str(ppv) + ','
  line += str(npv) + ',' + str(npd) + ',' + str(acc)
  return line
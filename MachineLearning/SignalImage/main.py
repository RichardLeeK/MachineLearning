import autoencoder as ae
import manifold as mf
import process as pc
import matplotlib.pyplot as plt
import datetime

if __name__ == '__main__':
  path = 'data/interpolate_gos/'
  signals, filenames = ae.load_data(path)
  gos_map = pc.gos_loader()
  total_image = []
  total_signal = []
  file_len_map = {}
 
  bef_cnt = 0
  for i in range(len(filenames)):
    filename = filenames[i].split('_')[0]
    imgs = ae.signal_to_img(signals[i])
    total_image.extend(imgs)
    total_signal.extend(signals[i])
    file_len_map[filename] = [bef_cnt, bef_cnt + len(imgs)]
    bef_cnt += len(imgs)
  """
  total_rep_imgs = ae.autoencoding_cnn(total_image, total_image, img_dim=128, encoding_dim=32)
  """
  gos_color = ['b', 'g', 'r', 'c', 'm']
  pen = open('img/label_gos.csv', 'w')
  cnt = 0
  cp_bef = datetime.datetime.now()
  for k, v in file_len_map.items():
    for i in range(v[0], v[1]):
      plt.figure(1)
      plt.imshow(total_image[i].reshape(128, 128))
      plt.savefig('img/signal_classification/ori/'+k+'_'+str(i)+'.png')
      plt.cla(); plt.clf()





      pen.write(str(i)+'\n')
      """
      plt.figure(1)
      plt.imshow(total_rep_imgs[i].reshape(128, 128))
      plt.savefig('img/signal_100/rep/'+k+'_'+str(i-v[0])+'.png')
      """
      if cnt % 10 == 0:
        cp_aft = datetime.datetime.now()
        print('P: ('+str(cnt)+'/'+str(len(total_image))+') ' + str(cp_aft-cp_bef))
        cp_bef = cp_aft
      cnt += 1

  """
  Y_o = mf.tSNELearning(total_image, n_component=5, init='pca', random_state=0)
  Y_r = mf.tSNELearning(total_rep_imgs, n_component=5, init='pca', random_state=0)
  
  fig = plt.figure(2)
  for k, v in file_len_map.items():
    for i in range(v[0], v[1]):
      plt.scatter(Y_o[i][0], Y_o[i][1], color=gos_color[gos_map[k]])
  fig2 = plt.figure(2)
  for k, v in file_len_map.items():
    for i in rnage(v[0], v[1]):
      plt.scatter(Y_r[i][0], Y_r[i][1], color=gos_color[gos_map[k]])
  plt.show()
  """
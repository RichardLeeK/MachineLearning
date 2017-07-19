from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

def create_mode(img_dim=128, nb_class=9):
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_dim, img_dim, 3)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(nb_class, activation='softmax'))
  model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
  return model
  
if __name__ == '__main__':
  path = ''
  
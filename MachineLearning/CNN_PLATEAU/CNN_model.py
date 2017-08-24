from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten
from keras.models import Model, Sequential
from keras.regularizers import l1,l2,l1_l2
import numpy as np
import dataGenerator as dg

def generate_cnn_autoencoder(x_range=900,y_range=110,encoded_dim=32):
    input_img = Input(shape=(y_range, x_range, 1),name='input_1')
    x = Conv2D(10, 5, activation='relu', padding='same', name='conv_1')(input_img)
    x = MaxPooling2D((2, 2), padding='same', name='pool_1')(x)
    encoded = Dense(encoded_dim,name='encoded')(x)
    
    x = UpSampling2D((2, 2))(encoded)
    decoded = Conv2DTranspose(1, 5, activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)

    return autoencoder
def train_encoder(model,img, env):
    optimizer=env.get_config("pre_train","optimizer",type="str")
    loss=env.get_config("pre_train","loss",type="str")
    epoch=env.get_config("pre_train","epoch",type="int")
    batch_size=env.get_config("pre_train","batch_size",type="int")
    verbose=env.get_config("system","verbose",type="int")
    memory_save=env.get_config("system","memory_save",type="int")

    model.compile(optimizer=optimizer, loss=loss)
    if memory_save==1:
        model.fit_generator(dg.encoder_x_generator(img,batch_size), steps_per_epoch=(len(img)/batch_size)+1, epochs=epoch, verbose=verbose, pickle_safe=False)
    else:
        model.fit(img, img, epochs=epoch, batch_size=batch_size, verbose=verbose, shuffle=False)
    return model

def reorganize_model(env):
    reg={"kernel":l1(.01),"bias":None,"activation":None}
    model=Sequential()
    model.add(Conv2D(10, 5, activation='relu', padding='same',input_shape=(110, 900, 1), name='conv_1'))
    model.add(MaxPooling2D((2, 2), padding='same', name='pool_1'))
    model.add(Dense(32,name='encoded'))
    
    model.add(Conv2D(10, 5, activation='relu', padding='same', 
                     kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"], activity_regularizer=reg["activation"]))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(10, 5, activation='relu', padding='same', 
                     kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"], activity_regularizer=reg["activation"]))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))

    h5_path=env.get_config("path","weight_load_path")
    model.load_weights(h5_path,by_name=True)

    return model

def train_model(model,trainX,trainY,env):
    optimizer=env.get_config("train","optimizer",type="str")
    loss=env.get_config("train","loss",type="str")
    epoch=env.get_config("train","epoch",type="int")
    batch_size=env.get_config("train","batch_size",type="int")
    verbose=env.get_config("system","verbose",type="int")
    memory_save=env.get_config("system","memory_save",type="int")

    model.compile(optimizer=optimizer, loss=loss)
    if memory_save==1:
        model.fit_generator(dg.x_generator(trainX,trainY,batch_size), steps_per_epoch=(len(trainX)/batch_size)+1, epochs=epoch, verbose=verbose, pickle_safe=False)
    else:
        model.fit(trainX, trainY, epochs=epoch, batch_size=batch_size, verbose=verbose, shuffle=False)
    return model

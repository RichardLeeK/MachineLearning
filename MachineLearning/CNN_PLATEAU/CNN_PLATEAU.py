import numpy as np
import gc

from env import Env
from env import int_input

import fileController as fc
import preprocess as pp
import preprocess_image as ppimg
import labelling as lb
import CNN_data as cd
import CNN_model as cm
import modelController as mc
import evaluator as ev
import cross_validation as cv

############main############
if __name__ == '__main__':
    env=Env()
    env.load_config("./config.ini")
    env=fc.get_path(env)
    skip_pre_train=env.get_config("system","skip_pre_train",type="int")
    memory_save=env.get_config("system","memory_save",type="int")

    # get train data
    datadict=fc.get_dataset(env.file["train_file_list"],feature=env.get_config("data","feature",type="list"))

    menu=int_input("0 : Test model / 1 : Train model / 2 : Cross-validation")

    if menu==0:
        model=mc.load_model(env)
        print("Evaluate Model")
        ev.eval_model(model,env)
    elif menu==1:
        print("Transform Data")
        datadict=pp.transfrom_dataset(datadict)
        window=env.get_config("CNN","window",type="int")
        y_range=env.get_config("CNN","y_range",type="int")
        step=env.get_config("CNN","step",type="int")
        encoded_dim=env.get_config("CNN","encoded_dim",type="int")

        print("Transform into image")
        imgdict=ppimg.transform_imgdict(datadict,window=window,y_range=y_range,step=step)

        print("Labeling")
        label_path=env.get_config("path","label_path")
        labeldata=lb.load_label(label_path)
        labeldict=lb.dataset_labeling(datadict,env.file["train_file_list"],labeldata)

        labeldict=lb.imgset_labeling(labeldict,window=window,y_range=y_range,step=step)

        print("Reshape data")
        trainX=cd.make_cnn_X_all(imgdict)
        trainY=cd.make_cnn_Y_all(labeldict)
       
        if memory_save==1:
            save_path=env.get_config("path","memory_save_path")
            np.save(save_path+"/data",trainX)
            gc.collect()
            trainX=np.load(save_path+"/data.npy",mmap_mode="r")
        if skip_pre_train==0:
          print("Generate Pre-training Model")
          model=cm.generate_cnn_autoencoder(x_range=window,y_range=y_range,encoded_dim=32)
    
          print("Pre-Training")
          model=cm.train_encoder(model,trainX,env)

          model.summary()
        
          mc.save_model(model,env)
        model=mc.load_model(env)
        print("Reorganize Model")
        model=cm.reorganize_model(env)
        
        model.summary()

        print("Train Model")
        model=cm.train_model(model,trainX,trainY,env)

        print("Save Model")
        mc.save_model(model,env)

        print("Evaluate Model")
        ev.eval_model(model,env)
    elif menu==2:
        print("Cross-validation")
        k=env.get_config("validation","fold_num",type="int")
        if k<=0:
            print("K value is Wrong")
        else:    
            print("Data preprocess & save")
            group=cv.divide_group(env.file["train_file_list"],k)
            print(group)
            train,test=cv.define_group(group,3,env)
            print(train)
            print(test)
############################
import preprocess as pp
import preprocess_image as ppimg
import CNN_data as cd
import numpy as np
from keras.models import model_from_json

from env import Env

def _refine_Y(Y,threshold=0.5): # for sigmoid
    for i in range(len(Y)):
        cur_y=Y[i]
        Y[i]= 1 if cur_y>threshold else 0
    return Y

def _decide_Y(Y): # for softmax
    result=[]
    for cur_y in Y:
        if cur_y[0]>cur_y[1]:
            result.append(0)
        else:
            result.append(1)
    return np.array(result)

def predict_data(data,model,env,pp=1):
    window=env.get_config("CNN","window",type="int")
    y_range=env.get_config("CNN","y_range",type="int")
    step=env.get_config("CNN","step",type="int")

    threshold=env.get_config("test","threshold",type="float")
    if pp==1:
        data=pp.transform_data(data)
    imgset=ppimg.transform_img(data,window=window,y_range=y_range,step=step)
    dataX=cd.make_cnn_X(imgset)
    pred=model.predict(dataX)
    total_len=len(data.T)
    pred=_refine_Y(pred,threshold=threshold)
    pred=translate_pred(pred,total_len,x_range=window,step=step)
    
    return pred

def translate_pred(pred,total_len,x_range=900,step=60):
    result=[0]*total_len
    plateau_progress=0
    for pidx in range(len(pred)):
        p=pred[pidx]
        if (p==0) and (plateau_progress==1):
          for idx in range((pidx-1)*step,(pidx-1)*step+x_range):
                if idx>=total_len:
                    continue
                result[idx]=1
          plateau_progress=0
        if p==1:
          if plateau_progress==0:
            plateau_progress=1
          else:
            for idx in range(pidx*step,pidx*step+x_range):
                if idx>=total_len:
                    continue
                result[idx]=1
    result=np.array(result)
    return result

def save_model(model,env):
    '''
    save trained model into json & h5

    Parameters
    ----------
    model : keras model
        trained model
    json_path : string
        filepath for model
    h5_path : string
        filepath for weight
    '''
    json_path=env.get_config("path","model_save_path")
    h5_path=env.get_config("path","weight_save_path")
    # serialize model to JSON
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5_path,overwrite="True")
    print("Saved model to disk")

def load_model(env):
    '''
    load trained model from json & h5

    Parameters
    ----------
    json_path : string
        filepath for model
    h5_path : string
        filepath for weight

    Returns
    -------
    model : keras model
        trained model
    '''
    json_path=env.get_config("path","model_load_path")
    h5_path=env.get_config("path","weight_load_path")
    # load json and create model
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_path)
    print("Loaded model from disk")
    model=loaded_model

    return model
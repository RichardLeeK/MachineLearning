import numpy as np

def make_cnn_X(imgset):
    result=[]
    for idx in range(len(imgset)):
        img=imgset[idx]
        result_img=np.array(img).astype('float32')/255
        result_img=np.reshape(result_img,(result_img.shape[0],result_img.shape[1],1))
        result.append(result_img)
    return np.array(result)

def make_cnn_X_all(imgdict):
    result=[]
    for setidx in range(len(imgdict)):
        imgset=imgdict[setidx]
        for idx in range(len(imgset)):
            img=imgset[idx]
            result_img=np.array(img).astype('float32')/255
            result_img=np.reshape(result_img,(result_img.shape[0],result_img.shape[1],1))
            result.append(result_img)
    return np.array(result)

def make_cnn_Y_all(labeldict):
    result=[]
    for setidx in range(len(labeldict)):
        labelset=labeldict[setidx]
        result.extend(labelset)
    return np.array(result)

def test_cnn_X(imgdict):
    result=dict()
    for setidx in range(len(imgdict)):
        imgset=imgdict[setidx]
        curimg=[]
        for idx in range(len(imgset)):
            img=imgset[idx]
            result_img=np.array(img).astype('float32')/255
            result_img=np.reshape(result_img,(result_img.shape[0],result_img.shape[1],1))
            curimg.append(result_img)
        result[setidx]=np.array(curimg)
    return result
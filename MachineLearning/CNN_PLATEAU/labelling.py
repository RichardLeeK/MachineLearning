import pandas as pd
import numpy as np

# load label file (=plateau time information)
def load_label(labelfile):
    labeldf=pd.read_csv(labelfile)
    labeldata=labeldf.as_matrix()
    return labeldata

# get plateau time information of one file
def get_label_time(labeldata,file_name):
    # file_name parsing
    start_idx=file_name.find("\\")+1
    end_idx=file_name.find("_")
    labeltime=np.empty((0,2))
    for label_row in labeldata:
        cur_row_name=label_row[0]
        if cur_row_name in file_name:
            labeltime=np.vstack((labeltime,label_row[2:4]))
    
    return labeltime

# labelling one file
def data_labeling(data,file_name,labeldata):
    # get time data
    labeltime=get_label_time(labeldata,file_name)
    
    label=[]
    time_lapse=data[0]
    for i in range(len(time_lapse)):
        time=time_lapse[i]
        is_pl=0
        for j in range(len(labeltime)):
            pl_start=labeltime[j][0]
            pl_end=labeltime[j][1]
            if pl_start<=time and pl_end>=time:
                is_pl=1
        label.append(is_pl)
    label=np.array(label)
    return label

# labelling all dataset
def dataset_labeling(datadict,filelist,labeldata):
    labeldict=dict()
    for i in range(len(datadict)):
        data=datadict[i]
        labeldict[i]=data_labeling(data,filelist[i],labeldata)
    return labeldict

def img_labeling(label,window=900,y_range=110,step=60):
    img_label=[]
    start_time=0
    while start_time<(len(label)-window):
        if (start_time+window)>=len(label):
            img_state=label[start_time:len(label)]
        else:
            img_state=label[start_time:start_time+window]
        if any(img_state):
            img_label.append(1)
        else:
            img_label.append(0)
        start_time=start_time+step
    return np.array(img_label)

def imgset_labeling(labeldict,window=900,y_range=110,step=60):
    img_labeldict=dict()
    for i in range(len(labeldict)):
        label=labeldict[i]
        img_label=img_labeling(label,window=window,y_range=y_range,step=step)
        img_labeldict[i]=img_label
    return img_labeldict
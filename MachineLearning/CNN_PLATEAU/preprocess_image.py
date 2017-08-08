import numpy as np

from env import Env

def fill(image,x_idx,y_idx,bound,value):
    if (x_idx<0) or (x_idx>=900):
        return image
    elif (y_idx<0) or (y_idx>=110):
        return image
    elif image[x_idx][y_idx]>=bound:
        return image
    else:
        image[x_idx][y_idx]=value
        return image
    
def fill_edge(image,x_idx,y_idx,value,bound,dist=1):
    fill(image,x_idx-dist,y_idx,bound,value)
    fill(image,x_idx-dist,y_idx-dist,bound,value)
    fill(image,x_idx-dist,y_idx+dist,bound,value)
    
    fill(image,x_idx+dist,y_idx,bound,value)
    fill(image,x_idx+dist,y_idx-dist,bound,value)
    fill(image,x_idx+dist,y_idx+dist,bound,value)
    
    fill(image,x_idx,y_idx-dist,bound,value)
    fill(image,x_idx,y_idx+dist,bound,value)

def transform_img(data,window=900,y_range=110,step=60):
    icps=np.int64(data[1])
    icps=np.array([icp for icp in icps if 0<icp<=y_range])
    image_set=[]
    start_time=0
    while start_time<(len(icps)-window):
        image=np.zeros((window,y_range), dtype=np.uint8)
        for time_idx in range(0,window):
            time=start_time+time_idx
            y_idx=icps[time]-1
            if y_idx<y_range:
                image[time_idx][y_idx]=255
            fill_edge(image,time_idx,y_idx,value=128,bound=255,dist=1)
        image_set.append(image.T)
        start_time=start_time+step
    return np.array(image_set)

def transform_imgdict(dataset,window=900,y_range=110,step=60):
    imgdict=dict()
    for i in range(len(dataset)):
        imgset=transform_img(dataset[i],window=window,y_range=y_range,step=step)
        imgdict[i]=imgset
    return imgdict
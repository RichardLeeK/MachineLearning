import pytz
import numpy as np
from datetime import datetime, timedelta

def fromOADate(v):
    return datetime(1899, 12, 30, 0, 0, 0, tzinfo=pytz.utc) + timedelta(days=v)

def get_group(data):
    timelapse=data[0]
    timelist=[]
    for i in range(len(timelapse)):
        curdt=fromOADate(timelapse[i])
        timelist.append(curdt)
    seclist=[]
    for i in range(len(timelist)):
        dt=timelist[i]
        seclist.append(dt.second)
    groups=[]
    cur_sec=0
    for i in range(len(seclist)):
        sec=seclist[i]
        if cur_sec!=sec:
            if cur_sec!=0:
                groups.append(np.array(cur_group))
            cur_sec=sec
            cur_group=[]
        cur_group.append(i)
    groups.append(np.array(cur_group))
    groups=np.array(groups)
    return groups

def group_by_sec(data,groups):
    new_data=[]
    for idxlist in groups:
        icp=0
        time=0
        for idx in idxlist:
            icp=icp+data[1][idx]
            time=time+data[0][idx]
        icp=icp/len(idxlist)
        time=time/len(idxlist)
        cur_data=[time,icp]
        new_data.append(np.array(cur_data))
    return np.array(new_data).T

# transform all data set
def transfrom_dataset(datadict):
    result=dict()
    for filenum in range(len(datadict)):
        data=datadict[filenum]
        cur_data=transform_data(data)
        result[filenum]=cur_data
    return result

def transform_data(data):
    groups=get_group(data)
    cur_data=group_by_sec(data,groups)
    return cur_data


def cut_idx_by_hour(data,hour_limit=3):
    timelapse=data[0]
    progress_hour=0
    cur_hour=0
    
    total_group=[]
    start_idx=0
    for idx in range(len(timelapse)):
        curdt=fromOADate(timelapse[idx])
        dthour=curdt.hour
        if cur_hour!=dthour:
            cur_hour=dthour
            progress_hour=progress_hour+1
            if progress_hour>hour_limit:
                end_idx=idx
                cur_group=np.arange(start_idx,end_idx)
                start_idx=idx
                
                total_group.append(cur_group)
    end_idx=idx+1
    cur_group=np.arange(start_idx,end_idx)
    total_group.append(cur_group)
    return total_group

def cut_by_hour(data,hour_limit=3):
    timelapse=data[0]
    progress_hour=0
    cur_hour=0
    
    total_data=[]
    start_idx=0
    for idx in range(len(timelapse)):
        curdt=fromOADate(timelapse[idx])
        dthour=curdt.hour
        if cur_hour!=dthour:
            cur_hour=dthour
            progress_hour=progress_hour+1
            if progress_hour>hour_limit:
                end_idx=idx
                sliced_data=data.T[start_idx:end_idx].T
                start_idx=idx
                
                total_data.append(sliced_data)
    end_idx=idx+1
    sliced_data=data.T[start_idx:end_idx].T
    total_data.append(sliced_data)
    return total_data

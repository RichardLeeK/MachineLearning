import numpy as np

def divide_group(filelist,K):
    N=len(filelist)
    group_size=int(N/K)+1
    sizeup_num=int(N%K)
    
    group_list=[]
    cur_group=[]

    start_idx=0

    for group_num in range(K):
        if group_num==sizeup_num:
            group_size=group_size-1
        cur_group=[]
        end_idx=start_idx+group_size
        cur_group=np.arange(start_idx,end_idx)
        group_list.append(np.array([filelist[idx] for idx in cur_group]))
        start_idx=end_idx

    group_list=np.array(group_list)
    return group_list

def define_group(group_list,seed,env):
    K=len(group_list)
    train=[]
    test=group_list[seed]
    count=0; idx=seed+1
    while count<K:
        if idx>=K:
            idx=0
        train.extend(group_list[idx])
        idx=idx+1
        if idx==seed:
            idx=idx+1
        count=count+1
    return train, test
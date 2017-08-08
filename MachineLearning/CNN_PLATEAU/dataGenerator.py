def encoder_x_generator(imgset,batch_size):
    while True:
        num=0
        while num<len(imgset):
            if (num+batch_size)>=len(imgset):
                end_idx=len(imgset)
            else:
                end_idx=num+batch_size
            batch_imgset=imgset[num:end_idx]
            num=num+batch_size
            yield batch_imgset, batch_imgset

def x_generator(imgset,label,batch_size):
    while True:
        num=0
        while num<len(imgset):
            if (num+batch_size)>=len(imgset):
                end_idx=len(imgset)
            else:
                end_idx=num+batch_size
            batch_imgset=imgset[num:end_idx]
            batch_label=label[num:end_idx]
            num=num+batch_size
            yield batch_imgset, batch_label
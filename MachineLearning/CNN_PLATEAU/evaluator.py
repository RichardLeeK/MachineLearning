import fileController as fc
import modelController as mc
import numpy as np
import labelling as lb
import preprocess as pp
import csv
import datetime
import matplotlib.pyplot as plt

def eval_model(model,env):
    testlist=env.file["test_file_list"]
    label_path=env.get_config("path","label_path")
    labeldata=lb.load_label(label_path)
    feature=env.get_config("data","feature",type="list")

    save_result=env.get_config("test","save_result",type="int")
    save_graph=env.get_config("test","save_graph",type="int")
    hour_limit=env.get_config("test","hour_limit",type="int")

    confusion=np.zeros(4)

    for fileidx in range(len(testlist)):
        filepath=testlist[fileidx]
        dataX=fc.get_data(filepath,feature)
        dataX=pp.transform_data(dataX)
        true=lb.data_labeling(dataX,filepath,labeldata)
        pred=mc.predict_data(dataX,model,env,pp=0)

        if save_graph==1:
            idx_group=pp.cut_idx_by_hour(dataX,hour_limit=3)
            figure_num=draw_graph(dataX,true,pred,fileidx,env,idx_group)        
        if save_result==1:
            confusion=confusion+get_confusion(true,pred)
    if save_result==1:
        confusion_result(confusion,env)

def draw_graph(data,true,pred,fileidx,env,idx_group=[]):
    figure_x=env.get_config("test","figure_x",type="int")
    figure_y=env.get_config("test","figure_y",type="int")
    figure_range=env.get_config("test","figure_range",type="int")

    graph_path=env.get_config("path","graph_path")

    if len(idx_group)==0:
        idx_group=[num for num in range(len(data))]

    figure_num=0
    for i in range(len(idx_group)):
        cur_group=idx_group[i]
        if i%figure_range==0:
            if figure_num>0:
                plt.savefig(graph_path+"/"+str(fileidx)+"_"+str(figure_num)+'.png')
            figure_num=figure_num+1
            plt.figure(figure_num,figsize=(figure_x,figure_y*figure_range))
        plt.subplot(figure_range*100+10+((i%figure_range)+1))
        timelapse=data[0][cur_group]
        icp=data[1][cur_group]
        cur_true=np.multiply(true[cur_group],max(icp))
        cur_pred=np.multiply(pred[cur_group],max(icp))
        
        plt.plot(timelapse,icp,'b',label="ICP")
        plt.plot(timelapse,cur_true.reshape(-1),'g',label="true")
        plt.plot(timelapse,cur_pred.reshape(-1),'r--',label="prediction")
        plt.grid()
        plt.legend()
    plt.savefig(graph_path+"/"+str(fileidx)+"_"+str(figure_num)+'.png')
    plt.clf()
    return figure_num

def confusion_result(confusion,env):
    tp=confusion[0]; tn=confusion[1]; fp=confusion[2]; fn=confusion[3]
    tpr=tp/(tp+fn)
    tnr=tn/(tn+fp)
    ppv=tp/(tp+fp)
    npv=tn/(fn+tn)
    acc=(tp+tn)/(tp+tn+fp+fn)
    netpred=(tpr+tnr)/2
    
    print("TP : "+str(tp))
    print("TN : "+str(tn))
    print("FP : "+str(fp))
    print("FN : "+str(fn))
    print("Sensitivity : "+str(tpr))
    print("Specificity : "+str(tnr))
    print("Net Prediction : "+str(netpred))
    print("PPV : "+str(ppv))
    print("NPV : "+str(npv))
    print("Accuracy : "+str(acc))
    
    save_confusion(np.array([tp,tn,fp,fn,tpr,tnr,netpred,ppv,npv,acc]),env)

def save_confusion(confusion,env):
    new_row=[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    new_row.extend(confusion)
    result_path=env.get_config("path","result_path")
    with open(result_path,'a') as f:
        writer=csv.writer(f)
        writer.writerow(new_row)

def get_confusion(true, pred):
    if len(true)!=len(pred):
        print("Wrong Input")
        return
    else:
        tp=0; tn=0; fp=0; fn=0
        for i in range(len(true)):
            if true[i]==pred[i]:
                if true[i]==0:
                    tn=tn+1
                else:
                    tp=tp+1
            else:
                if pred[i]==0:
                    fn=fn+1
                else:
                    fp=fp+1    
        return np.array([tp,tn,fp,fn])
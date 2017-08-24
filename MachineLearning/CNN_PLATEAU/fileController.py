import os
import pandas as pd
import numpy as np

def get_real_name(filename):
    return filename[filename.find("\\")+1:filename.find("_AR.csv")]

def load_and_store(filelist,env):
    feature=env.get_config("data","feature",type="list")
    save_path=env.get_config("path","memory_save_path")

    # error check
    if len(filelist)<=0:
        print("You have to set the filelist")
        return

    # load data and store
    for filenum in range(len(filelist)):
        file=filelist[filenum]
        data=get_data(file,feature)
        name=get_real_name(file)
        print(len(data.T))
        print(name)
        np.save(save_path+"/"+name,data)


def get_dataset(filelist,feature=['datetime','icp']):
    """    
    Load csv files from filelist
    if feature is set, data will be filtered

    Parameters
    ----------
    filelist : array
        Filepath array
    feature : array
        Selected column name array

    Returns
    -------
    datadict : dictionary {filenum : matrix(feature x timelapse)}
        Loaded data dictionary
    """
    # error check
    if len(filelist)<=0:
        print("You have to set the filelist")
        return

    # fill data dictionary
    datadict=dict()
    for filenum in range(len(filelist)):
        file=filelist[filenum]
        datadict[filenum]=get_data(file,feature)

    return datadict

def get_data(filename,feature=['datetime','icp']):
    """
    Load csv file from filename
    if feature is set, data will be filtered

    Parameters
    ----------
    filename : string
        Filepath String
    feature : array
        Selected column name array

    Returns
    -------
    datamat : matrix (feature x timelapse)
        Loaded data matrix
    """
    # error check
    if filename=="":
        print("You have to set the file name")
        return

    # load csv file by using pandas
    df=pd.read_csv(filename)
    # when filter is set
    if len(feature)>0:
        df=df_filtering(df,feature)
    data=df.as_matrix().T
 
    return data

def df_filtering(df,feature=[]):
    """ 
    Select the dataframe column by looking up feature
    NaN value will be erased

    Parameters
    ----------
    df : dataframe
        Original Pandas DataFrame
    feature : array
        Selected column name array

    Returns
    -------
    result_df : dataframe
        Selected DataFrame
    """
    # selected index = column indexes having feature name
    selected_idx=[]
    # get column names from dataframe
    df_colname=df.columns.values

    
    for cur_f in feature:
        for i in range(len(df_colname)):
            cur_col=df_colname[i]
            cur_col=cur_col.lower()
            if cur_col==cur_f:
                selected_idx.append(i)
                break;
    '''
    # Find index of feature 
    for i in range(len(df_colname)):
        cur_col=df_colname[i]
        cur_col=cur_col.lower()
        # compare cur_col & col_name in feature
        for col_name in feature:
            # if dataframe column has one string of feature array
            if col_name in cur_col:
                selected_idx.append(i)
                break;
    '''
    result_df=df.iloc[:,selected_idx].dropna()
    return result_df

def get_filelist(dirname):
    """ 
    Get filepath list in input directory

    Parameters
    ----------
    dirname : string
        File Directory
    
    Returns
    -------
    filelist : array
        list of filepath string
    """
    # get all file path in directory
    filenames = os.listdir(dirname)
    filelist=[]

    # concatenate dirname + filename
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        filelist.append(full_filename)
    
    return filelist

def get_path(env):
    trainlist=get_filelist(env.get_config("path","training_path"))
    testlist=get_filelist(env.get_config("path","test_path"))

    env.file["train_file_list"]=trainlist
    env.file["test_file_list"]=testlist

    return env
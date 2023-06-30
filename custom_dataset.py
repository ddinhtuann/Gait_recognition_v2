import os
import time
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   
import tensorflow as tf
from random import randint

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, paths, y_onehot, n_chunk=15, batch_size=4, input_size=(15,512), shuffle=True):
        'Initialization'
        self.paths = paths
        self.y_onehot = y_onehot
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_classes = y_onehot.shape[1]
        self.shuffle = shuffle
        self.on_epoch_end()
        # print(self.label_IDs)
        # print(self.label_enc.inverse_transform([0,43]))
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paths) / self.batch_size))



    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # print("+++++LEN INDEX = ", len(indexes))
        # assert len(indexes) == self.batch_size
        k = self.batch_size - len(indexes)
        while k:
            indexes = np.append(indexes,indexes[randint(0,k-1)])
            k = k - 1

        # print(indexes.shape)
        # print(self.list_paths)

        # Find list of paths
        list_paths_batch = [self.paths[k] for k in indexes]
        P = []
        #Init X,y of batch
        X = np.empty((self.batch_size, *self.input_size))

        y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        # Generate data
        for i, p in enumerate(list_paths_batch):
            # Store sample
            P.append(p)
            X[i,] = np.load(p)
            
            # Store class
            y[i] = self.y_onehot[indexes[i]] 

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

def getAllData(train_path, test_path=None, train_ex_path=None):
    label_enc = LabelEncoder()
    
    #get train data
    train_df = pd.read_csv(train_path,dtype=str,header=None)
 
    if train_ex_path:
        df_ex = pd.read_csv(train_ex_path,dtype=str,header=None)
        for i in range(50):
            train_df =pd.concat([train_df,df_ex], ignore_index=True)

    train_df =  train_df.sample(frac=1)

    X_train = np.asarray([np.load(p) for p in train_df[0]])

    #get train label
    y_train = label_enc.fit_transform(train_df[1])

    np.save(f"labels.npy",label_enc.classes_)

    if test_path:
        #get train data
        df = pd.read_csv(test_path,dtype=str,header=None)
        test_df =  df.sample(frac=1)

        X_test = np.asarray([np.load(p) for p in test_df[0]])

        #get train label
        y_test = label_enc.transform(test_df[1])
    else:
        return X_train, y_train
    
    return X_train, y_train, X_test, y_test


def getAllData_v3(train_path, trainex_path, n_chunk):
    #read csv data
    df = pd.read_csv(train_path, dtype=str, header=None)
    dfex = pd.read_csv(trainex_path, dtype=str, header=None)
    dfex =  dfex.sample(frac=1)

    #
    list_paths = df[0].values.tolist() + dfex[0].values.tolist()*6
    list_pIDs = df[1].values.tolist() + dfex[1].values.tolist()*6

    #get label
    label_enc = LabelEncoder()
    label_IDs = label_enc.fit_transform(list_pIDs)
    n_classes = len(label_enc.classes_)

    np.save(f"gait_labels/labels_n{n_chunk}.npy",label_enc.classes_)

    y_onehot = tf.keras.utils.to_categorical(label_IDs, num_classes=n_classes)
    return list_paths, y_onehot


def getAllData_v2(train_path, test_path=None, train_ex_path=None):
    print("getAllData.........")
    bname = os.path.splitext(train_path)[0].split("_")[-1]
    n_chunk = int(bname[1:])
    print("n_chunk", n_chunk)
    label_enc = LabelEncoder()
    

    #get train data
    train_df = pd.read_csv(train_path,dtype=str,header=None)
 
    if train_ex_path:
        df_ex = pd.read_csv(train_ex_path,dtype=str,header=None)
        for i in range(6):
            train_df =pd.concat([train_df,df_ex], ignore_index=True)

    # train_df =  train_df.sample(frac=1)
    X_train = [np.load(p) for p in train_df[0]] #+ [np.load(p) for p in df_ex[0]]*6
    X_train = np.asarray(X_train)


    #get train label
    y_train = label_enc.fit_transform(train_df[1])

    np.save(f"gait_labels/labels_n{n_chunk}.npy",label_enc.classes_)

    if test_path:
        #get train data
        test_df = pd.read_csv(test_path,dtype=str,header=None)
        test_df =  test_df.sample(frac=1)

        X_test = np.asarray([np.load(p) for p in test_df[0]])

        #get train label
        y_test = label_enc.transform(test_df[1])
    else:
        return X_train, y_train, X_train, y_train
    
    
    return X_train, y_train, X_test, y_test


def getExData(train_ex_path=None):

    #get train data
    train_df = pd.read_csv(train_ex_path,dtype=str,header=None) 
    
    # train_df =  train_df.sample(frac=1)

    X_train = np.asarray([np.load(p) for p in train_df[0]])

    #get train label
    y_train = train_df[1]
    
    return X_train, y_train, train_df[0]

def get_Exdata_path(train_ex_path=None):
    train_df = pd.read_csv(train_ex_path, dtype=str, header = None)
    return train_df[0]    

if __name__ == '__main__':
    data_gen = DataGenerator("features_data.csv",n_classes=5,input_size=(15,512))
    X, y = data_gen.getAllData()
    print(X.shape)
    idx = np.argmax(y,axis=1)
    print(data_gen.label_enc.inverse_transform(idx))

    
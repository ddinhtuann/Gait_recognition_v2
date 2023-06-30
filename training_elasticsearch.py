import argparse
import glob
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.multiprocessing as mp
import tqdm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from call_index_search import call_index
from custom_dataset import getAllData_v2, getExData
from gait_model import Gait_Recognition
from processed_data import Processed_Data

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
  except RuntimeError as e:
    print(e)

def checkDatabase(feat_dir, n_chunk):
    #check dir to save features
    if os.path.isdir(feat_dir):
        print(f"{feat_dir} is already existed!!!")
        return None
    else:
        os.makedirs(feat_dir)

    pIDs = os.listdir("segmented_data")

    proc_data = Processed_Data(n_chunks=n_chunk)
    for pID in tqdm.tqdm(pIDs,desc="Extracting base feature"):
        gIDs = os.listdir(os.path.join('segmented_data',pID))
        gIDs.sort()

        for gID in gIDs:
            seg_paths = glob.glob(os.path.join('segmented_data',pID,gID,"*"))
            seg_paths.sort()
  
            proc_data(seg_paths,pID,isTrain=True)
            #extract_base_features          
            proc_data.extract_base_features(feat_dir,seg_paths,gID)
    print("Done extract database!!!")


def finetuneMultiGait(person_dir, n_chunk):
    #directory of extend data
    seg_dir = f"gait_segmentation/segmented_data_ex"
    feat_dir = f"gait_features/features_data_ex_n{n_chunk}"
    os.makedirs('gait_csv',exist_ok=True)
    os.makedirs('gait_labels',exist_ok=True)


    #Check database
    checkDatabase(f"gait_features/features_data_n{n_chunk}", n_chunk)

    #get ex person ID
    with open('exIDs.txt','r') as f:
        pIDs = f.read().splitlines()

    #get query ID
    with open('queryIDs.txt','r') as f:
        queryIDs = f.read().splitlines()
    
    for id in queryIDs:
        pIDs.append(id)
        shutil.rmtree(os.path.join(seg_dir,id),ignore_errors=True)
        shutil.rmtree(os.path.join(feat_dir,id),ignore_errors=True)
    
    proc_data = Processed_Data(n_chunks=n_chunk, seg_dir=seg_dir, feat_dir=feat_dir)

    # #remove old dir
    # shutil.rmtree(seg_dir,ignore_errors=True)
    # print(f"removed {seg_dir}")
    # shutil.rmtree(feat_dir,ignore_errors=True)
    # print(f"removed {feat_dir}")

    csv_path = f"gait_csv/train_ex_n{n_chunk}.csv"
    if os.path.isfile(csv_path):
        os.remove(csv_path)
    print(f"removed {csv_path}")
    # shutil.rmtree(feat_dir,ignore_errors=True)

    # os.makedirs(feat_dir,exist_ok=True)
    # os.makedirs('gait_csv',exist_ok=True)

    fi = open(csv_path, "w")
    for pID in tqdm.tqdm(pIDs,desc="Extracting feature"):
        if os.path.exists(os.path.join(feat_dir,pID)):
            feat_paths = glob.glob(os.path.join(feat_dir,pID,"*.npy"))
            # print(feat_paths)
            for p in feat_paths:
                fi.write(f"{p},{pID}\n")
            
            print(f"Done for {pID}")
            # exit(0)
            continue
        
        print(f"Extracing {pID}")
        gIDs = os.listdir(os.path.join(person_dir,pID))
        gIDs.sort()

        os.makedirs(os.path.join(feat_dir,pID))

        for gID in gIDs:       
            img_paths = glob.glob(os.path.join(person_dir,pID,gID,"*"))
            img_paths.sort()

            proc_data(img_paths,pID,isTrain=True)

            #segment image
            seg_paths = proc_data.createSegmentID(mode="gray", thres_ratio=0.0)

            #extract_features          
            proc_data.extract_features_v2(seg_paths,gID,fi)
    
    torch.cuda.empty_cache()
    fi.close()
    print("Done feature extracting!")

    gait_model = Gait_Recognition()
    
    #Callbacks
    checkpoint_filepath = 'models/checkpoint'
    earlyStopping = EarlyStopping(monitor='loss', patience=5, verbose=0, mode='min')
    model_checkpoint_callback = ModelCheckpoint(
        filepath=f"gait_models/gait_model_latest_n{n_chunk}.h5",
        monitor='loss',
        mode='min',
        save_best_only=True)

    #params for training
    params = {"lr":0.00125,
        "epoch": 80,
        # "batch_size":32,
        "batch_size":256,
        "callbacks" : [ earlyStopping, model_checkpoint_callback]
        }

    #create data for training
    X_train, y_train, X_test, y_test = getAllData_v2(f"gait_csv/train_n{n_chunk}.csv",None,f"gait_csv/train_ex_n{n_chunk}.csv")

    print(X_train.shape)
    print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    labels = np.load(f"gait_labels/labels_n{n_chunk}.npy",allow_pickle=True)

    n_classes = len(labels)
    print("number of classes:", n_classes)

    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

    #training
    print('Start training...')
    gait_model.build_model(input_shape=X_train.shape[1:], n_classes=n_classes)
    gait_model.train(X_train, y_train_onehot,X_test, y_test_onehot, **params)
    print('finishlamlt')


    #Convert features (None, 7(15), 512) dimension to (None, 64) dimension
    X_ex_train, y_ex_train, train_paths = getExData(f"gait_csv/train_ex_n{n_chunk}.csv")
    gait_extractor = gait_model.get_extractor()
    ex_train_pred = gait_extractor.predict(X_ex_train)
    
    call_index(ex_train_pred, y_ex_train, n_chunk)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser for predict Gait')
    # parser.add_argument('--clear_cache', default=False, action='store_true',help='clear feature data and train_ex.csv')
    parser.add_argument('--clear_data', default=False, action='store_true',help='clear cache data of n chunks')
    parser.add_argument('--clear', default=False, action='store_true',help='clear training data')


    
    parser.add_argument('-n', type=int,help='number of images of a chunk', required=True)


    args = parser.parse_args()


    ###CLEAR TRAINING DATA
    if args.clear:
        data_dir = "training_images"
        dirs = os.listdir(data_dir)
        dirs = list(filter(lambda x: False if "ex" in x else True,dirs))
        print("Removing ",*dirs)
        for pID in dirs:
            shutil.rmtree(os.path.join(data_dir,pID),ignore_errors=True)

        print("Done clear training images!")
        exit(0)


    ###CLEAR BASEDATA
    if args.clear_data:
        data_dir = f"gait_features/features_data_n{args.n}"
        shutil.rmtree(data_dir,ignore_errors=True)
        print(f"removed {data_dir}")

        csv_path = f"gait_csv/train_n{args.n}.csv"
        if os.path.isfile(csv_path):
            os.remove(csv_path)
        print(f"removed {csv_path}")

        exit(0)    

    finetuneMultiGait("training_images",args.n)




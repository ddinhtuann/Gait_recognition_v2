import os
import tqdm
import cv2
import glob
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import torch
import torch.multiprocessing as mp

from custom_dataset import  getAllData
from gait_model import Gait_Recognition
from processed_data import Processed_Data
import argparse

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
  except RuntimeError as e:
    print(e)

def finetuneMultiGait(person_dir):
    pIDs = os.listdir(person_dir)

    seg_dir = "segmented_data_ex"
    feat_dir = "features_data_ex"
    
    proc_data = Processed_Data(n_chunks=7, seg_dir=seg_dir, feat_dir=feat_dir)
    #remove temp dir
    shutil.rmtree(seg_dir,ignore_errors=True)



    # shutil.rmtree(feat_dir,ignore_errors=True)

    # try:
    #     os.remove("train_ex.csv")
    # except OSError:
    #     pass

    os.makedirs(feat_dir,exist_ok=True)

    for pID in tqdm.tqdm(pIDs,desc="Extracting feature"):
        gIDs = os.listdir(os.path.join(person_dir,pID))
        gIDs.sort()

        # save_path = os.path.join(feat_dir,f"{pID}_{gIDs[0]}_0-{proc_data.n_chunks-1}.npy")
        # if os.path.exists(save_path):
        #     continue 
        
        for gID in gIDs:
            img_paths = glob.glob(os.path.join(person_dir,pID,gID,"*"))

            proc_data(img_paths,pID,isTrain=True)

            #segment image
            seg_paths = proc_data.createSegmentID(mode="gray")

            #extract_features          
            proc_data.extract_features(seg_paths,gID)
    
    torch.cuda.empty_cache()

    n_chunks = 7 #number of sequential frames
    gait_model = Gait_Recognition()
    
    #Callbacks
    checkpoint_filepath = 'models/checkpoint'
    earlyStopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')
    model_checkpoint_callback = ModelCheckpoint(
        filepath="gait_model_latest.h5",
        monitor='loss',
        mode='min',
        save_best_only=True)

    #params for training
    params = {"lr":0.001,
        "epoch":80,
        # "batch_size":32,
        "batch_size":512,
        "callbacks" : [ earlyStopping, model_checkpoint_callback]
        }

    #create data for training
    X_train, y_train, X_test, y_test = getAllData("train.csv","test.csv","train_ex.csv")

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    labels = np.load("labels.npy",allow_pickle=True)
    n_classes = len(labels)

    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

    #training
    print('Start training...')
    gait_model.build_model(input_shape=X_train.shape[1:], n_classes=n_classes)
    gait_model.train(X_train, y_train_onehot,X_test, y_test_onehot,**params)
    print('finishlamlt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser for predict Gait')
    parser.add_argument('--clear_cache', default=False, action='store_true',help='clear feature data and train_ex.csv')
    parser.add_argument('--model_type', default=False,help='clear feature data and train_ex.csv')
    args = parser.parse_args()


    os.system("rmdir /s features_data_ex /q")

    os.remove("train_ex.csv")

    finetuneMultiGait("training_images")




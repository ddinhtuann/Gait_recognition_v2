from itertools import groupby
from torchreid.utils import FeatureExtractor
from person_segment_api import Person_Seg_API

import os, glob
import cv2
import numpy as np
import shutil
import tqdm
import time

class Processed_Data():
    """docstring for Processed_Data"""
    def __init__(self,n_chunks=7):

        self.n_chunks = n_chunks

        #load segment model
        self.seg_model = Person_Seg_API()
        
        #load torchreid extraction
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='torchreid_pretrained/osnet_x1_0_imagenet.pth',
            # model_path="torchreid_pretrained/osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth",
            device='cuda'
        )
        
        #Fake init
        self.extractor(['fake.png'])


    def __call__(self, img_paths, person_ID="0000", isTrain=True):
        img_paths.sort()
        self.img_paths = img_paths
        self.person_ID = person_ID
        self.isTrain = isTrain
        

    def createSegmentID(self,mode = "gray"):


        person_dir = os.path.join("segmented_data_ex",self.person_ID)
        os.makedirs(person_dir,exist_ok=True)

        #create gait dir
        gIDs = os.listdir(person_dir)
        gait_dir = os.path.join(person_dir,f"g{len(gIDs):04}")
        os.makedirs(gait_dir,exist_ok=False)

        seg_paths = []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        for p in self.img_paths:
            img = cv2.imread(p)
            seg_img = self.seg_model.segment_person(img)

            if mode == "gray":
                seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
                seg_img = clahe.apply(seg_img)

            bname = os.path.basename(p)
            save_path = os.path.join(gait_dir,bname)
            cv2.imwrite(save_path,seg_img)
            seg_paths.append(save_path)
            # fi.write(save_path+"\n")

        # fi.close()

        return seg_paths

    def extract_features(self, seg_paths,gID=None):
        
        #get features from segmented img
        features = self.extractor(seg_paths)
        features = features.cpu()

        if not  self.isTrain:
            return features

        #create features data dir
        feat_dir = "features_data_ex"

        n,m = features.shape
        n_chunks = self.n_chunks
        #save sample for tranning 
        st = 0 
        en = n_chunks 
         
        fi = open(f"train_ex.csv", "a")

        while en <= n:
            sample = features[st:en,...]
            save_path = os.path.join(feat_dir,f"{self.person_ID}_{gID}_{st}-{en-1}.npy")
            np.save(save_path, sample)
            #write npy path to file
            fi.write(f"{save_path},{self.person_ID}\n")
            st = st + 1
            en = en + 1

        fi.close()
from operator import index
import os
import glob
import numpy as np
import argparse
import time
from call_index_search import call_search

# import torch
from processed_data import Processed_Data
from gait_model import Gait_Recognition
from custom_dataset import get_Exdata_path
from tqdm import tqdm

import tensorflow.keras.backend as K
import tensorflow as tf

import torch
import time
import json
from elastic_search import search

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
  except RuntimeError as e:
    print(e)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def process_result(result,r):
    for d in result:
        if d['id'] == r['id']:
            if d['score'] < r['score']:
                d['score'] = r['score']
                d['chunk'] = r['chunk']
                d['chunk_train'] = r['chunk_train']
                d['train_path'] = r['train_path']
            d['cnt'] = d['cnt'] + 1
            return None
    result.append(r)

def predictGait(img_dir):
    img_paths = glob.glob(os.path.join(img_dir,"*"))
    img_paths.sort()
    
    proc_data(img_paths, isTrain=False)
    n_chunk = proc_data.n_chunks

    st1 = time.time()
    #segment image
    seg_paths = proc_data.createSegmentID(mode="gray",thres_ratio=0.001)
    # print("Time segment", time.time()-st1)

    #extract_features
    st2 = time.time()
    feat = proc_data.extract_features(seg_paths)
    torch.cuda.empty_cache()
    # print("Time extract", time.time()-st2)

    st3 = time.time()

    gait_extractor = gait_model.get_extractor()
    # print(gait_extractor.summary())

    train_paths = get_Exdata_path(f"gait_csv/train_ex_n{n_chunk}.csv")
    
    chunks = gait_model.get_chunks(feat)
    pred = gait_extractor.predict(chunks)
    

    result = []
    print("Begin searching")
    for c in range(pred.shape[0]):
        start = time.time()

        res = call_search(query_vector= pred[c], n_chunk=n_chunk)
        
        stt, id = res['_id'].split("_")
        score = 1000 - res['_score'] 
        
        r = {}
        pID,gID,c_t = os.path.basename(train_paths[int(stt)]).split("_")

        r['chunk'] = str(c)
        r['chunk_train'] = c_t.split("-")[0]
        r['train_path'] = os.path.join("F:\Gait_recognition_v2\training_images", pID, gID)
        r['n'] = str(n_chunk)
        r['score'] = str(score)
        r['id'] = str(id)
        r['cnt'] = 1
        
        process_result(result, r)
    
    return result 

def predictMulti(img_dirs):
    result = []
    for img_dir in tqdm(img_dirs):

        print(img_dir)
        r = predictGait(img_dir)
        for item in r:
            result.append(item)
        # result.append(r)

    return json.dumps(result) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser for predict Gait')

    parser.add_argument('-j', type=str, help='json of img directories', default = 'sample_multi.json')
    parser.add_argument('-n', type=int, help='n_chunk', default= 15)



    args = parser.parse_args()

    n_chunk = args.n
    gait_model = Gait_Recognition()
    gait_model.load_model(f"gait_models/gait_model_latest_n{n_chunk}.h5", f"gait_labels/labels_n{n_chunk}.npy")
    proc_data = Processed_Data(n_chunks=args.n,seg_dir='gait_segmentation/test_segment_ex')

    with open(args.j,'r') as f:
        img_dirs = json.load(f)
    
    print("data###", predictMulti(img_dirs))
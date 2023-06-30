import os
import glob
import numpy as np
import argparse
import time

# import torch
from processed_data import Processed_Data
from gait_model import Gait_Recognition
from custom_dataset import getExData
from tqdm import tqdm

import tensorflow.keras.backend as K
import tensorflow as tf

import torch
import time
import json

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

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
            if d['score'] > r['score']:
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

    X_train, y_train, train_paths = getExData(f"gait_csv/train_ex_n{n_chunk}.csv")
    
    train_pred = gait_extractor.predict(X_train)
    chunks = gait_model.get_chunks(feat)
    pred = gait_extractor.predict(chunks)
    

    result = []
    for c in range(pred.shape[0]):
      
        dist = euclidean_distance([pred[c], train_pred])
        idx = np.argmin(dist)

        r = {}
        pID,gID,c_t = os.path.basename(train_paths[idx]).split("_")
        score = round(dist[idx].numpy()[0],2)

        r['chunk'] = str(c)
        r['chunk_train'] = c_t.split("-")[0]
        r['train_path'] = os.path.join(r"C:\Pro\rcs\face\production\251\tracking\Gait_recognition\training_images", pID, gID)
        r['n'] = str(n_chunk)
        r['score'] = str(score)
        r['id'] = pID
        r['cnt'] = 1

        process_result(result,r)
        
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

    # data = ['test_imgs_ex/p0021', 'test_imgs_ex/p0022/g0009','test_gait','test_imgs_ex2/Hai_p0023/g0001','test_imgs_ex2/Hai_p0023/g0002','test_imgs_ex2/Tung_p0024/g0001']

    # with open(args.j,'w') as f:
    #     json.dump(data,f)

    with open(args.j,'r') as f:
        img_dirs = json.load(f)

    #img_dirs = ["testing_images/aotrang"]
    print("data###",predictMulti(img_dirs))

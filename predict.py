import os
import glob
import numpy as np
import argparse
import time

# import torch
from processed_data import Processed_Data
from gait_model import Gait_Recognition
from custom_dataset import getExData

import tensorflow.keras.backend as K
import tensorflow as tf

import torch
import time
import json

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gait_model = Gait_Recognition()
gait_model.load_model("gait_model_latest.h5","labels.npy")
proc_data = Processed_Data()


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
    
    proc_data(img_paths, isTrain=False)

    st1 = time.time()
    #segment image
    seg_paths = proc_data.createSegmentID(mode="gray")
    # print("Time segment", time.time()-st1)

    #extract_features
    st2 = time.time()
    feat = proc_data.extract_features(seg_paths)
    torch.cuda.empty_cache()
    # print("Time extract", time.time()-st2)

    st3 = time.time()

    gait_extractor = gait_model.get_extractor()
    # print(gait_extractor.summary())

    X_train, y_train, train_paths = getExData("train_ex.csv")
    
    train_pred = gait_extractor.predict(X_train)
    chunks = gait_model.get_chunks(feat)
    pred = gait_extractor.predict(chunks)

    result = []
    for c in range(pred.shape[0]):
        dist = euclidean_distance([pred[c],train_pred])
        idx = np.argmin(dist)

        r = {}
        pID,gID,c_t = os.path.basename(train_paths[idx]).split("_")
        score = round(dist[idx].numpy()[0],2)

        r['chunk'] = str(c)
        r['chunk_train'] = c_t.split("-")[0]
        r['train_path'] = os.path.join(r"C:\Pro\rcs\face\production\251\tracking\Gait_recognition\training_images",pID,gID)
        r['n'] = str(7)
        r['score'] = str(score)
        r['id'] = pID
        r['cnt'] = 1

        process_result(result,r)
    return json.dumps(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser for predict Gait')

    parser.add_argument('--img_dir', type=str, help='images folder of person', required=False)
    parser.add_argument('--model_type', type=str, help='images folder of person', required=False)

    args = parser.parse_args()


#     print("ATHANG:", predictGait("test_imgs_ex/p0021"))
#     print("AOTRANG:", predictGait("test_imgs_ex/p0022/g0009"))
#     print("P0008:", predictGait("test_gait"))
#     print("HAINM:", predictGait("test_imgs_ex2/Hai_p0023/g0001"))
#     print("HAINM:", predictGait("test_imgs_ex2/Hai_p0023/g0002"))
#     print("TUNGDM:", predictGait("test_imgs_ex2/Tung_p0024/g0001"))



    print("data###",predictGait(args.img_dir))





    
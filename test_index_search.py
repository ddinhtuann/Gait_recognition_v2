from dataclasses import dataclass
from custom_dataset import getExData
from gait_model import Gait_Recognition
import os
from call_index_search import call_index, call_search
import glob
from processed_data import Processed_Data
from elastic_search import encode_array
import json 


def test_search(img_dir):
    img_paths = glob.glob(os.path.join(img_dir,"*"))
    img_paths.sort()
    proc_data(img_paths, isTrain=False)

    seg_paths = proc_data.createSegmentID(mode="gray",thres_ratio=0.001)
    feat = proc_data.extract_features(seg_paths)
    chunks = gait_model.get_chunks(feat)
    pred = gait_extractor.predict(chunks)
    



if __name__ == "__main__":
    n_chunk =15
    gait_model = Gait_Recognition()
    #proc_data = Processed_Data(n_chunks=n_chunk,seg_dir='gait_segmentation/test_segment_ex')
    gait_model.load_model(f"gait_models/gait_model_latest_n{n_chunk}.h5", f"gait_labels/labels_n{n_chunk}.npy")
    
    X_ex_train, y_ex_train, path = getExData(f"gait_csv/train_ex_n{n_chunk}.csv")
    gait_extractor = gait_model.get_extractor()
    ex_train_pred = gait_extractor.predict(X_ex_train)
    
    print("Call index api")
    call_index(ex_train_pred, y_ex_train)
    
    
    
    
    

    


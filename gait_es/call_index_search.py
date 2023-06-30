from elastic_search import encode_array, index_all, create_index, index_one, search, check_index
import numpy as np
from gait_model import Gait_Recognition
import requests
from processed_data import Processed_Data
import glob
import os
import json
from tqdm import tqdm


def convert_to_base64(vectors):
    ls_base64 = []
    if vectors.ndim == 1:
        ls_base64 = encode_array(vectors)
        return ls_base64
    else:
        for vector in vectors:
            str = encode_array(vector)
            ls_base64.append(str)
        return ls_base64


def call_index_api(train_pred, ids_train):
    model_endpoint = 'http://10.0.68.100:5000/index_one'
    data = []
    for x,id in zip(train_pred, ids_train):
        v = {
                "id" : id,
                "vector_b64": encode_array(x)
        }
        data.append(v)
    
    data_json = json.dumps(data)
    print(data_json)
    r = requests.post(url = model_endpoint, json = data_json)
    assert r.status_code == 200
    return r.json()

def call_search_api(query_vector, index_name, number = None):
    model_endpoint = 'http://10.0.68.100:5000/search_vectors'
    query_base64 = convert_to_base64(query_vector)
    data = {'query_vectors': query_base64, 'index_name': index_name, 'number': number}

    r = requests.post(url=model_endpoint, data=data)
    assert r.status_code == 200
    return r.json()


def call_index(train_pred, ids_train):
    
    if not check_index(index_name = "gait_index"):
        create_index(name= "gait_index")
    else:
        os.system("curl -X DELETE http://127.0.0.1:9200/gait_index")
        create_index(name="gait_index")
    
    for stt, (x, id) in tqdm(enumerate(zip(train_pred, ids_train))):
        v = {
                "stt": stt,
                "id": id,
                "vector_b64": encode_array(x)
        }
        
        index_all(vector= v, name= "gait_index")

    
def call_search(query_vector):
    query_base64 = convert_to_base64(query_vector)
    
    res =  search(name= "gait_index", query_vector=query_base64, number=1)
    return  res['hits']['hits'][0]

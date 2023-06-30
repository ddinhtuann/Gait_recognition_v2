from elasticsearch import Elasticsearch, NotFoundError
import base64
import numpy as np
import time
from tqdm import tqdm

_float32_dtype = np.dtype('>f4')

def decode_float_list(base64_string):
    buffer = base64.b64decode(base64_string)
    return np.frombuffer(buffer, dtype=_float32_dtype).tolist()

def encode_array(arr):
    base64_str = base64.b64encode(np.array(arr).astype(_float32_dtype)).decode("utf-8")
    return base64_str

es = Elasticsearch('http://127.0.0.1:9200/', send_get_body_as='POST', retry_on_timeout=True, timeout=5000)

def create_index(name) :     
    request_body = {
        "settings": {
            "number_of_shards": 5,
            "number_of_replicas": 1
            },

        "mappings": {

                "properties": {
                    
                    "embedding_vector": {
                        "type": "dense_vector",
                        "dims": 64
                        }
                    }
                }
    
        }
    
    es.indices.create(index = name, body = request_body)
    print(f"Created {name} index... {request_body['mappings']['properties']}")

def index_one(vector, name):
    body = {
        "id": vector["id"],
        "embedding_vector": decode_float_list(vector["vector_b64"])
        }
    es.index(index= name, body=body)
    #print("Index successfully")


def index_all(vector, name):

        body = {
            "stt": vector["stt"],
            "id": vector["id"],
            "embedding_vector": decode_float_list(vector["vector_b64"])
        }
        es.index(index= name, body= body)
    #print("Indexing successfully")

def searchv2(name, query_vector, number):
    
    body_name = {
        "query": {
            "function_score": {
                "boost_mode": "replace",
                "script_score": {
                    "script": {
                        "source": "binary_vector_score",
                        "lang": "knn",
                        "params": {
                            "cosine": True,
                            "field": "embedding_vector",
                            "vector": decode_float_list(query_vector)
                        }
                    }
                }
            }
        },
        "size": number
    }
    return es.search(index = name, body = body_name)


def search(name, query_vector, number):
    query = {
        "query": {
            "function_score": {
                "boost_mode": "replace",
                "script_score": {
                    "script": {
                        "source": " 1 + dotProduct(params.vector, 'embedding_vector') ",
                            
                        "params": {
                            "vector": decode_float_list(query_vector)
                            }
                        }
                    }
                }
            },
        "size": number
        }

    return es.search(index = name, body = query)


def check_index(index_name):
    return es.indices.exists(index= index_name)

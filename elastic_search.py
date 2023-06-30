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

es = Elasticsearch('http://10.0.68.100:9200/', send_get_body_as='POST', retry_on_timeout=True, timeout=5000)
es1 = Elasticsearch('http://127.0.0.1:9200/', send_get_body_as='POST', retry_on_timeout=True, timeout=5000)

def create_index(name) :     
    request_body = {
        "settings": {
            "number_of_shards": 3,
            "number_of_replicas": 0
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

def create_index_binary(name):
    request_body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
            },
        "mappings": {

                "properties": {
                    "embedding_vector": {
                        "type": "binary",
                        "doc_values": True
                        }
                    }
                }
    
        }

    es1.indices.create(index=name, body= request_body)
    print(f"Created {name} index... {request_body['mappings']['properties']}")

def index_binary(index_name, vector, pid):
    body = {
        "embedding_vector" : vector["vector_b64"]
    }
    es1.index(index=index_name, body=body, id=pid)


def index_one(index_name, vector, pid):
    body = {
        "embedding_vector": decode_float_list(vector["vector_b64"])
        }
    es.index(index= index_name, body=body, id = pid)
    #print("Index successfully")

def search_binary(name, query_vector, number):
    
    body_name = {
        "query": {
            "function_score": {
                "boost_mode": "replace",
                "script_score": {
                    "script": {
                        "source": "binary_vector_score",
                        "lang": "knn",
                        "params": {
                            "cosine": False,
                            "field": "embedding_vector",
                            "vector": decode_float_list(query_vector)
                        }
                    }
                }
            }
        },
        "size": number
    }
    return es1.search(index = name, body = body_name)


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

def searchv2(name, query_vector, number):
    query = {
    "query": {
        "function_score": {
            "boost_mode": "replace",
            "script_score": {
                "script": {
                    "source": "cosineSimilarity(params.vector, 'embedding_vector') + 1.0 ",
                        
                    "params": {
                        "vector": decode_float_list(query_vector)
                        }
                    }
                }
            }
        }
    }
    
    return es.search(index = name, body = query, size =number)

def searchv3(name, query_vector, number):
    query = {
    "query": {
        "function_score": {
            "boost_mode": "replace",
            "script_score": {
                "script": {
                    "lang": "javascript",
                    "source": """
                    float[] v = doc['embedding_vector'].vectorValue;
                    float l2 = 0;
                    for (int i = 0; i < v.length; i++) {
                        l2 += Math.pow((v[i] - params.vector[i]), 2);
                    }
                    return 1000 - Math.sqrt(l2);
                    """, 
                    "params": {
                        "vector": decode_float_list(query_vector)
                        }
                    }
                },
            
            }
        }
    }
    return es.search(index = name, body = query, size =number)



def searchv4(name, query_vector, number):
    query = {
    "query": {
        "function_score": {
            "boost_mode": "replace",
            "script_score": {
                "script": {
                    "source": "1000 - l2norm(params.vector, 'embedding_vector')", 
                    "params": {
                        "vector": decode_float_list(query_vector)
                        }
                    }
                },
            
            }
        }
    }

    return es.search(index = name, body = query, size =number)




def check_index(index_name):
    return es.indices.exists(index= index_name)

def check_index_binary(index_name):
    return es1.indices.exists(index= index_name)

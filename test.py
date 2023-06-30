from PIL import Image
import io
import requests
from gait_model import Gait_Recognition
from custom_dataset import getExData
from call_index_search import call_index
def call_model(file_path):
    '''Send an input image through the network.'''
    model_endpoint = 'http://0.0.0.0:6001/face_renovation_predict_img'

    with open(file_path, 'rb') as file:
        file_form = {'image': (file_path, file, 'image/png')}
        r = requests.post(url=model_endpoint, files=file_form)
        assert r.status_code == 200
        im = Image.open(io.BytesIO(r.content))
        return im

if __name__ == "__main__":

    gait_model = Gait_Recognition()
    gait_model.load_model(f"gait_models/gait_model_latest_n{15}.h5", f"gait_labels/labels_n{15}.npy")


    #Convert features (None, 7(15), 512) dimension to (None, 64) dimension
    X_ex_train, y_ex_train, train_paths = getExData(f"gait_csv/train_ex_n{15}.csv")
    gait_extractor = gait_model.get_extractor()
    ex_train_pred = gait_extractor.predict(X_ex_train)
    

    #create document and index data
    call_index(ex_train_pred.tolist(), y_ex_train)
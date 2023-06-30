
import tensorflow as tf
import numpy as np
import tqdm
import json

from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import schedules



class Gait_Recognition():
    def __init__(self):
        self.model = None
        self.input_shape = None
        self.output_shape = None
        self.labels = None

    def load_model(self,model_path,labels_path):
        self.labels = np.load(labels_path,allow_pickle=True)
        self.model = keras.models.load_model(model_path)

        #fake init
        n,f = self.model.input.shape[1:]
        self.model.predict(tf.random.uniform((1,n,f)))
        
        print("Done load model:",model_path)

    def get_extractor(self):
        return Model(self.model.inputs, self.model.layers[-7].output) #-7 for out64  | -4 for out32


    def build_model(self,input_shape=(7,512), n_classes=20):
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape))
        model.add(Dropout(0.5))

        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(n_classes))
        model.add(Activation('softmax'))

        self.model = model
        self.input_shape = input_shape
        self.output_shape = n_classes


    def train(self,X_train,y_train,X_val,y_val,lr=0.001,epoch=20,batch_size=32, callbacks=[]):
        #complie model
        model = self.model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr),metrics=['accuracy'])
#         model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1,shuffle=True, validation_data=(X_val, y_val),callbacks=callbacks)
        model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1,shuffle=True,callbacks=callbacks)

        # save_path = f"gait_pretrained/gait_lr{lr}_epoch{epoch}_bz{batch_size}_class{self.output_shape}.h5"
        # model.save(save_path, overwrite=True)
        # model.save("gait_model_latest.h5", overwrite=True)
        # print("Done! Model saved at gait_model_latest.h5")

    def predict(self,pID_feat):
       
        n,m = pID_feat.shape
        n_chunks = self.model.input.shape[1]
        print(n, m)
        st = 0 
        en = n_chunks #7

        samples = []
        while en <= n:
            sample = pID_feat[st:en,...]
            sample = np.expand_dims(sample,axis=0)
            samples.append(sample)
            st = st + 1
            en = en + 1


        samples = np.vstack(samples)
        print(samples.shape)

        pred = self.model.predict(samples)
        return pred

    def get_chunks(self,pID_feat):
       
        n,m = pID_feat.shape
        
        n_chunks = self.model.input.shape[1]

        st = 0 
        en = n_chunks #7

        print(f"n={n}, n_chunks={n_chunks}")
        assert n >= n_chunks

        samples = []
        while en <= n:
            sample = pID_feat[st:en,...]
            sample = np.expand_dims(sample,axis=0)
            samples.append(sample)
            st = st + 1
            en = en + 1


        samples = np.vstack(samples)

        return samples

    def test(self,test_feats):
        '''
            test_feats(numpy array): array of sequential gait e.g: (18,512)
        '''      

        preds = []
        for pID in tqdm.tqdm(test_feats,desc="Predicting"):
            pID_feat = test_feats[pID]
            pred = self.predict(pID_feat)
            preds.append(pred)
    
        return np.asarray(preds)



if __name__ == "__main__":

    params = {"lr":0.001,
              "epoch":20,
              "batch_size":32,
             }

    gait_model = Gait_Recognition()
    gait_model.build_model()
    # gait_model.train(None,None,None,None,**params)

import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model
import time

class Person_Seg_API():
    def __init__(self,checkpoint = 'Unet_2020-07-20'):
        self.model = create_model(checkpoint)
        self.model.eval()  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.model.to(self.device)
        self.transform = albu.Compose([albu.Normalize(p=1)], p=1)
        
        #fake init
        self.model(torch.rand((1,3,288,128)).to(self.device))

    def segment_person(self,image):
        
        
       
        padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
        x = self.transform(image=padded_image)["image"]
        x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

        x = x.to(self.device)

        with torch.no_grad():
            prediction = self.model(x)[0][0]

        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)

        #count ratio of 1 on img
        h,w = mask.shape
        ratio = np.count_nonzero(mask)/(h*w)
        mask = 255*(1.0-mask)
        mask = np.expand_dims(mask, axis=2)
        mask = np.tile(mask, (1,1,3))

        mask = mask.astype(np.uint8)
        result = cv2.add(image, mask)

        return  result, ratio


if __name__ == '__main__':

    model = Person_Seg_API()
    # print(model.summary())
    for i in range(10,12):
        img = cv2.imread(f'real{i}.png')
        # img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = model.segment_person(img)

        cv2.imwrite(f"output{i}.jpg",mask)
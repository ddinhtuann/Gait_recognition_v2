import os
import cv2
import shutil
from itertools import groupby
from random import randint

def split_train_test():
    with open("featu.csv","r") as f:
        img_paths = f.read().splitlines()

    pIDs = [list(i) for j, i in groupby(img_paths, lambda p: p.split("_")[1])]

    for p in pIDs:
        gIDs = [list(i) for j, i in groupby(p, lambda p: p.split("_")[2])]

        v = randint(0, len(gIDs)-1)
        for i in range(len(gIDs)):
            if i == v:
                file_name = "test.csv"
            else:
                file_name = "train.csv"

            with open(file_name,"a") as f:
                for l in gIDs[i]:
                    f.write(l + "\n")


def convert_rgb2gray(img_dir):
    for root, dirs, files in os.walk(img_dir, topdown=False):
        save_dir = root.replace("test_segmented_ex","test_segmented_ex_gray")
        os.makedirs(save_dir,exist_ok=True)

        for n in files:
            path = os.path.join(root,n)
            img = cv2.imread(path,0)
            
            save_path = os.path.join(save_dir,n)       
            cv2.imwrite(save_path,img)

convert_rgb2gray("test_segmented_ex")

            





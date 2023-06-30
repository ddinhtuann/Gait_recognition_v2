import os, glob
import shutil
from tqdm import tqdm

#ROOT/person_id/gait_id/img_name

root = "WARD"
dest = "images"
img_paths = os.listdir(root)

for p in tqdm(img_paths):
    save_dir = os.path.join(dest,p[:4],p[4:8])
    os.makedirs(save_dir,exist_ok=True)

    src = os.path.join(root,p)
    dst = os.path.join(save_dir,p)
    shutil.copyfile(src, dst)




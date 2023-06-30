import os
import glob
import shutil
def re_id_directory(src_dir):
    count = 1
    for file_path in os.listdir(src_dir):
        src_path =  os.path.join(src_dir, file_path)
        dst_name = f"ex{count:03}"
        dst_path = os.path.join(src_dir, dst_name)
        os.rename(src_path, dst_path)
        count += 1

def add_gID(src_dir):
    for pID_path in os.listdir(src_dir):

        gID = os.path.join(src_dir, pID_path, "g0000")
        if not os.path.exists(gID):
            os.makedirs(gID, exist_ok= True)

        for img_path in os.listdir(os.path.join(src_dir, pID_path)):

            if os.path.isfile(os.path.join(src_dir, pID_path, img_path)):
                shutil.move(os.path.join(src_dir, pID_path, img_path), gID)



if __name__ == '__main__':
    src_dir = "ex_train_images"
    re_id_directory(src_dir)
    add_gID(src_dir)
    exIDs = f"exIDS.txt"
    if os.path.isfile(exIDs):
        os.remove(exIDs)
    print(f"removed {exIDs}")

    fi = open(exIDs, "w")

    for file_path in os.listdir(src_dir):
        fi.write(f"{file_path}\n")



    



    


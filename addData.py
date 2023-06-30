import os
import shutil
import argparse

def copyData2Traning(person_ID, src_dir):
    dst = "training_images"
    gIDs = []

    try:
        gIDs = os.listdir(os.path.join(dst,person_ID))
    except OSError:
        pass

    gID_dir = os.path.join(dst,person_ID,f"g{len(gIDs):04}")
    # os.makedirs(gID_dir,exist_ok=False)

    shutil.copytree(src_dir, gID_dir)

    with open("queryIDs.txt", 'w') as f:
        f.write(person_ID)
        
    print('finishlamlt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser for addData')

    parser.add_argument('--person_id', type=str, help='person ID', required=True)
    parser.add_argument('--img_dir', type=str, help='images folder of person', required=True)
    parser.add_argument('--model_type', type=str, help='images folder of person', required=False)

    args = parser.parse_args()
    # copyData2Traning("BacPT","images_ex3/BacPT/g0002")
    copyData2Traning(args.person_id, args.img_dir)
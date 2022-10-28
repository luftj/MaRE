# e 16b runtime index construction TÜDR200

import os, shutil
from time import time
import json

if __name__ == "__main__":
    sheets = "E:/data/tüdr200/preussen/blattschnitt/blattschnitt.geojson"
    images_list = "E:/data/tüdr200/preussen/list.txt"
    annotations = "E:/data/tüdr200/preussen/annotations.csv"

    # save old config
    shutil.move("config.py", "config.py.old")
    shutil.copy("experiments/config_e16b.py", "config.py")

    data_dir = "E:/data/tüdr200/preussen/"

    # os.makedirs(f"E:/experiments/idx_e16b/",exist_ok=True)
    
    t_start = time()

    # build index
    cmd = f"""python indexing.py --rebuild {sheets}"""
    os.system(cmd)

    t_end = time()

    time_taken = t_end-t_start

    print("time:",time_taken)

    with open(sheets) as fr:
        data = json.load(fr)
        num_sheets = len(data["features"])

    print("#sheets:", num_sheets)
    print("time per sheet:",time_taken/num_sheets)
    
    shutil.move("config.py.old", "config.py")
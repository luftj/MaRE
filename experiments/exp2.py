import os, shutil
from tkinter import E

from eval_scripts.plot_index import plot_single_distribution

sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
images_list = "E:/data/deutsches_reich/osm_baseline/list.txt"
data_dir = "data/osm_baseline/"
out_dir = "E:/experiments/e2/"

os.makedirs(out_dir, exist_ok=True)
os.makedirs("E:/experiments/idx_kdr100/index", exist_ok=True)

# check for data  # to do
# create data  # to do
# make_osm_baseline.py {sheets} {images_list} {data_dir}

# set config
shutil.move("config.py", "config.py.old")
shutil.copy("experiments/config_e2.py", "config.py")

try:
    # check for index
    # index_exists = len(os.listdir("E:/experiments/idx_kdr100/index/keypoints")) > 0
    # print("index present:", index_exists)
    
    # run georeferencing
    restrict_hypos=0
    cmd = f""" python main.py {images_list} {sheets} -r {restrict_hypos}"""# --crop"""# --noimg
    os.system(cmd)

    # run eval scripts

    # annotations = "/e/data/deutsches_reich/SBB/cut/annotations.csv"
    # py -3.7 eval_georef.py {annotations} {sheets} # <- funktioniert nicht für baseline

    # make figures  # to do
    # georef scores

finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")
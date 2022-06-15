import os, shutil
from tkinter import E

from eval_scripts.plot_index import plot_single_distribution

sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
images_list = "E:/data/deutsches_reich/osm_baseline/list.txt"
data_dir = "data/osm_baseline/"
out_dir = "E:/experiments/e1/"

os.makedirs(out_dir, exist_ok=True)
os.makedirs("E:/experiments/idx_kdr100/index", exist_ok=True)

# check for data  # to do
# create data  # to do
# make_osm_baseline.py {sheets} {images_list} {data_dir}

# set config
shutil.move("config.py", "config.py.old")
shutil.copy("experiments/config_e1.py", "config.py")

try:
    # check for index
    # index_exists = len(os.listdir("E:/experiments/idx_kdr100/index/keypoints")) > 0
    # # index_exists = False
    # print("index present:", index_exists)

    # # create index
    # cmd = f"""python indexing.py --list {images_list} {"--rebuild" if not index_exists else ""} {sheets}"""
    # os.system(cmd)
    # shutil.move("index_result.csv", out_dir + "index_result.csv")
    
    # run georeferencing
    restrict_hypos=20
    cmd = f""" python main.py {images_list} {sheets} -r {restrict_hypos}"""# --noimg
    # cmd = "python main.py E:/data/deutsches_reich/osm_baseline/10.png E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson -r 30 --gt 10"
    # os.system(cmd)

    # run eval scripts
    cmd = "python eval_logs.py"
    os.system(cmd)
    shutil.move("eval_result.csv", f"{out_dir}/eval_result.csv")

    # annotations = "/e/data/deutsches_reich/SBB/cut/annotations.csv"
    # py -3.7 eval_georef.py {annotations} {sheets}

    # # make figures  # to do
    # # indexing scores
    # from eval_scripts.plot_index import plot_single_distribution
    # scores = []
    # with open(f"{out_dir}/index_result.csv") as fr:
    #     fr.readline()
    #     for line in fr:
    #         gt,idx = line.split(" : ")
    #         scores.append(int(idx))
    #         os.makedirs(f"{out_dir}/figures", exist_ok=True)
    # plot_single_distribution(scores,f"{out_dir}/figures")

    # retrieval scores
    # georef scores

    # from eval_logs import plot_score_dist
    # scores = []
    # with open(f"{out_dir}/eval_result.csv") as fr:
    #     fr.readline()
    #     for line in fr:
    #         vals = line.split("; ")
    #         score = vals[7]
    #         scores.append(score)
    # plot_score_dist(scores)


finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")
import os, shutil

from eval_scripts.plot_index import plot_single_distribution
from experiments.eval_baseline_georef import calc_and_dump
from experiments.summary_baseline_georef import load_results,filter_results,summary_and_fig

sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
images_list = "E:/data/deutsches_reich/osm_baseline/list.txt"
data_dir = "E:/data/deutsches_reich/osm_baseline/"
out_dir = "E:/experiments/e2/"

os.makedirs(out_dir, exist_ok=True)
os.makedirs("E:/experiments/idx_kdr100/index", exist_ok=True)

try:
    # check for data
    os.makedirs(data_dir, exist_ok=False)
except OSError:
    print(f"data already present")
else:
    cmd = f"make_osm_baseline.py {sheets} {images_list} {data_dir}"
    os.system(cmd)

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
    shutil.move("eval_result.csv", f"{out_dir}/eval_result.csv")

    # run eval scripts
    calc_and_dump(sheets, out_dir)

    # make figures for georef scores
    results = load_results(out_dir+"baseline_georef_scores.csv")
    results = filter_results(results, "E:/experiments/e1/eval_result.csv")
    with open(f"{out_dir}/registration_summary.txt","w") as outfile:
        mean, median = summary_and_fig(results, out_dir, outfile=outfile)

finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")
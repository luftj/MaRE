# experiment 12b: case study TÜDR200
# Auflage: Preußische Landesaufnahme 1902-x (ohne Gitterlinien)

from math import ceil
import os, shutil, glob

sheets = "E:/data/tüdr200/blattschnitt/blattschnitt.geojson"
images_list = "experiments/output/list.txt"
annotations = "experiments/output/annotations.csv"

# save old config
shutil.move("config.py", "config.py.old")
shutil.copy("experiments/config_e12b_preu.py", "config.py")

from indexing import search_list

from experiments.exp8 import make_register_figs
from experiments.exp8 import get_all_results
from eval_logs import eval_logs
from eval_georef import summary_and_fig

data_dir = "experiments/output/"
out_dir = "E:/experiments/e12b_preu/"
idx_basepath = "E:/experiments/idx_tudr200_preu/"
os.makedirs(out_dir,exist_ok=True)

try:
    # check for index
    os.makedirs(f"{idx_basepath}/index/keypoints",exist_ok=True)
    index_exists = len(os.listdir(f"{idx_basepath}/index/keypoints")) > 0
    print("index present:", index_exists, f"{idx_basepath}/index/keypoints")
    if not index_exists:
        # build index
        cmd = f"""python indexing.py --rebuild {sheets}"""
        os.system(cmd)

    # eval index ranks
    if os.path.isfile(f"{out_dir}/index_result.csv"):
        print("indexing reults already present")
    else:
        lps = search_list(images_list)
        with open(f"{out_dir}/index_result.csv","w") as fp:
            for sheet,rank in lps:
                fp.write("%s : %d\n"%(sheet,rank))
        
    # determine necessary number of spatial verifications
    restrict_hypos = 0
    with open(f"{out_dir}/index_result.csv") as fp:
        for line in fp:
            sheet,rank = line.split(" : ")
            rank=int(rank)
            if rank > restrict_hypos:
                restrict_hypos = rank

    max_hypos = 30
    restrict_hypos = ceil(restrict_hypos/5)*5 # round to next higher step of 5
    restrict_hypos = max(restrict_hypos,5) # leave some room for errors :)
    restrict_hypos = min(restrict_hypos,max_hypos)
    print(f"TÜDR200: verifying {restrict_hypos} hypotheses")

    # run georeferencing
    if len(glob.glob(f"{out_dir}/2022*.log")) < 1: # to do: check something smarter here
        cmd = f""" python main.py {images_list} {sheets} -r {restrict_hypos} --isize 6000"""# --noimg
        os.system(cmd)
    else:
        print("georeferencing has already been run")
    
    # run evaluation scripts
    # evaluate retrieval hit rate
    if os.path.isfile(f"{out_dir}/eval_result.csv"):
        print("already evaluated retrieval")
    else:
        eval_logs(out_dir)
        shutil.move("eval_result.csv", f"{out_dir}/eval_result.csv")
    
    # evaluate georef accuracy
    if os.path.isfile(f"{out_dir}/eval_georef_result.csv"):
        print("already evaluated georef")
        # print("appending...")
        # with open(f"{out_dir}/georef_summary.txt","w") as outfile:
        #     summary_and_fig(annotations, sheets, outfile=outfile, append_to=f"{out_dir}/eval_georef_result.csv")
    else:
        with open(f"{out_dir}/georef_summary.txt","w") as outfile:
            summary_and_fig(annotations, sheets, outfile=outfile, downscale_factor=3, debug_plot=True)
        shutil.move("eval_georef_result.csv", f"{out_dir}/eval_georef_result.csv")
        shutil.move("georef_error.png", f"{out_dir}/georef_error.png")
    exit()

    # make summary and figs
    percent_correct, mean_index, median_index, max_rank = get_all_results(out_dir)
    with open(f"{out_dir}/retrieval_summary.txt","a") as outfile:
        outfile.write(f"mean index: {mean_index}\n")
        outfile.write(f"median index: {median_index}\n")
        outfile.write(f"max index: {max_rank}\n")
    make_register_figs(out_dir, n_bins=5, y_step=1)

finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")
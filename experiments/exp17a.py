# experiment 17c: runtime KDR100 without early termination heuristic, without spatial verification hypos

import os, shutil, glob
from experiments.exp8 import get_all_results

restrict_hypos = 1

if __name__ == "__main__":
    sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
    images_list = "E:/data/deutsches_reich/SLUB/cut/raw/list.txt"
    annotations = "E:/data/deutsches_reich/SLUB/cut/raw/annotations.csv"

    # save old config
    shutil.move("config.py", "config.py.old")
    shutil.copy("experiments/config_e17a.py", "config.py")

    from eval_logs import eval_logs

    data_dir = "E:/data/deutsches_reich/SLUB/cut/raw/"
    out_dir = "E:/experiments/e17a/"
    os.makedirs(out_dir,exist_ok=True)

    try:
        print(f"KDR100: verifying {restrict_hypos} hypotheses")

        # run georeferencing
        if len(glob.glob(f"{out_dir}/2022*.log")) < 1: # to do: check something smarter here
            cmd = f""" python main.py {images_list} {sheets} -r {restrict_hypos} --noimg"""
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
        
        # make summary and figs
        percent_correct, mean_index, median_index, max_rank = get_all_results(out_dir)
        with open(f"{out_dir}/retrieval_summary.txt","a") as outfile:
            outfile.write(f"mean index: {mean_index}\n")
            outfile.write(f"median index: {median_index}\n")
            outfile.write(f"max index: {max_rank}\n")

    finally:
        # reset config
        os.remove("config.py")
        shutil.move("config.py.old", "config.py")
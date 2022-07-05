# Experiment 7: Robustheit gegen unterschiedliche Signaturklassen

from math import ceil
import os, shutil, glob

from eval_logs import eval_logs
from experiments.eval_baseline_georef import calc_and_dump

from experiments.exp3 import make_figure, get_all_results

sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
images_list_base = "E:/data/deutsches_reich/osm_baseline/list.txt"

# save old config
shutil.move("config.py", "config.py.old")

symbol_types = ["streets","railways"]
data_dir_base = "E:/data/osm_baseline_signatures/"
out_dir_base = "E:/experiments/e7/"

try:
    # make degraded dataset
    for nv in symbol_types:
        shutil.copy(f"experiments/config_e7_{nv}.py", "config.py") # set config

        data_dir=f"{data_dir_base}{nv}/"
        try:
            # check for data
            os.makedirs(data_dir, exist_ok=False)
        except OSError:
            print(f"data for {nv} already present")
        else:
            cmd = f"make_osm_baseline.py {sheets} {images_list_base} {data_dir}"
            os.system(cmd)

    # for nv in symbol_types:
    #     shutil.copy(f"experiments/config_e7_{nv}.py", "config.py") # set config
    #     # check for index
    #     os.makedirs(f"E:/experiments/idx_kdr100_{nv}/index/keypoints",exist_ok=True)
    #     index_exists = len(os.listdir(f"E:/experiments/idx_kdr100_{nv}/index/keypoints")) > 0
    #     # index_exists = False
    #     print("index present:", index_exists, f"E:/experiments/idx_kdr100_{nv}/index/keypoints")
    #     if index_exists:
    #         continue
    #     # create index
    #     images_list=f"{data_dir_base}{nv}/list.txt"
    #     cmd = f"""python indexing.py --list {images_list} "--rebuild" {sheets}"""
    #     os.system(cmd)
    #     out_dir=f"{out_dir_base}{nv}/"
    #     os.makedirs(out_dir,exist_ok=True)
    #     shutil.move("index_result.csv", out_dir + "index_result.csv")

    for nv in symbol_types:
        shutil.copy(f"experiments/config_e7_{nv}.py", "config.py") # set config
        out_dir=f"{out_dir_base}{nv}/"
        if os.path.isfile(f"{out_dir}/index_result.csv"):
            print("index already present")
            continue
        images_list=f"{data_dir_base}{nv}/list.txt"
        os.makedirs(out_dir,exist_ok=True)

        # cmd = f"""python indexing.py --list {images_list} {sheets}"""
        # os.system(cmd)
        # shutil.move("index_result.csv", out_dir + "index_result.csv")
        from indexing import search_list
        lps = search_list(images_list)
        with open(f"{out_dir}/index_result.csv","w") as fp:
            for sheet,rank in lps:
                fp.write("%s : %d\n"%(sheet,rank))
        
    for nv in symbol_types:
        shutil.copy(f"experiments/config_e7_{nv}.py", "config.py") # set config
        images_list=f"{data_dir_base}{nv}/list.txt"
        out_dir=f"{out_dir_base}{nv}/"
        os.makedirs(out_dir,exist_ok=True)

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
        print(f"symbol class {nv}: verifying {restrict_hypos} hypotheses")

        if len(glob.glob(f"{out_dir_base}/{nv}/2022*.log")) < 1: # to do: check something smarter here
            # run georeferencing
            cmd = f""" python main.py {images_list} {sheets} -r {restrict_hypos}"""# --noimg
            os.system(cmd)
            for file in glob.glob(f"{out_dir_base}/tmp/*"):
                shutil.move(file,out_dir)
            os.rmdir(f"{out_dir_base}/tmp/")
        else:
            print("georeferencing has already been run for", nv)
            continue
    
    # run evaluation scripts
    for nv in symbol_types:
        shutil.copy(f"experiments/config_e7_{nv}.py", "config.py") # set config
        # evaluate retrieval hit rate
        out_dir=f"{out_dir_base}{nv}/"
        os.makedirs(out_dir, exist_ok=True)
        
        if os.path.isfile(f"{out_dir}/eval_result.csv"):
            print("already evaluated retrieval")
            continue
       
        eval_logs(out_dir)
        shutil.move("eval_result.csv", f"{out_dir}/eval_result.csv")
    
    for nv in symbol_types:
        shutil.copy(f"experiments/config_e7_{nv}.py", "config.py") # set config
        # evaluate georef accuracy
        out_dir=f"{out_dir_base}/{nv}/"
        
        if os.path.isfile(f"{out_dir}/baseline_georef_scores.csv"):
            print("already evaluated georef")
            continue

        calc_and_dump(sheets, out_dir)

    # compare different runs
    results = []
    results.append(get_all_results("E:/experiments/e2/", "Flüsse")) # add baseline as first result
    for nv in symbol_types:
        out_dir=f"{out_dir_base}{nv}/"
        labels = {
            "streets":"Straßen",
            "railways":"Bahnlinien"}
        results.append(get_all_results(out_dir, labels[nv]))
    
    # make comparison figure
    make_figure(results, out_dir_base, "Signatureklasse", max_error=1000, x_type="string")

finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")
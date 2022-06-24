# Experiment 4: Robustheit gegen Occlusion/Verdeckung/fehlende Signaturen

from math import ceil
import os, shutil, glob

from eval_logs import eval_logs
from experiments.eval_baseline_georef import calc_and_dump
from experiments.summary_retrieval_logs import make_summary
from experiments.summary_baseline_georef import load_results,filter_results,summary_and_fig

from experiments.exp3 import make_figure, get_all_results

sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
images_list_base = "E:/data/deutsches_reich/osm_baseline/list.txt"

# set config
shutil.move("config.py", "config.py.old")
shutil.copy("experiments/config_e4.py", "config.py")

occlusion_params = [300,500,700]
data_dir_base = "E:/data/osm_baseline_degraded/occlusion/"
out_dir_base = "E:/experiments/e4/"

try:
    # make degraded dataset
    for nv in occlusion_params:
        data_dir=f"{data_dir_base}{nv}/"
        try:
            # check for data
            os.makedirs(data_dir, exist_ok=False)
        except OSError:
            print(f"data for {nv} already present")
        else:
            cmd = f"make_osm_baseline.py {sheets} {images_list_base} {data_dir} --circles {nv}"
            os.system(cmd)

    for nv in occlusion_params:
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
        
    # determine necessary number of spatial verifications
    for nv in occlusion_params:
        images_list=f"{data_dir_base}{nv}/list.txt"
        out_dir=f"{out_dir_base}{nv}/"
        os.makedirs(out_dir,exist_ok=True)

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
        print(f"occlusion amount {nv}: verifying {restrict_hypos} hypotheses")

        # restrict_hypos=20
        if len(os.listdir(f"{out_dir_base}/{nv}")) < 1000:
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
    for nv in occlusion_params:
        # evaluate retrieval hit rate
        out_dir=f"{out_dir_base}{nv}/"
        os.makedirs(out_dir, exist_ok=True)
        
        if os.path.isfile(f"{out_dir}/eval_result.csv"):
            print("already evaluated retrieval")
            continue
       
        eval_logs(out_dir)
        shutil.move("eval_result.csv", f"{out_dir}/eval_result.csv")
    for nv in occlusion_params:
        # evaluate georef accuracy
        out_dir=f"{out_dir_base}/{nv}/"
        
        if os.path.isfile(f"{out_dir}/baseline_georef_scores.csv"):
            print("already evaluated georef")
            continue

        calc_and_dump(sheets, out_dir)

    # compare different runs
    results = []
    results.append(get_all_results("E:/experiments/e2/", 0)) # add baseline as first result
    for idx,nv in enumerate(occlusion_params):
        data_dir=f"{data_dir_base}{nv}/"
        
        with open(f"{data_dir}/occlusion.txt") as fp:
            vals = []
            for line in fp:
                sheet,occl_val = line.strip().split(",")
                vals.append(float(occl_val))
        percent_occlusion = sum(vals)/len(vals)
        
        out_dir=f"{out_dir_base}{nv}/"
        results.append(get_all_results(out_dir, percent_occlusion))
    
    # make comparison figure
    make_figure(results, out_dir_base, "Verdeckung", max_error=500)

finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")
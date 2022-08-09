# Experiment 5: Robustheit gegen unterschiedliche Bildgröße

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
shutil.copy("experiments/config_e5.py", "config.py")

test_params = [200,500]
data_dir_base = "E:/data/osm_baseline/"
out_dir_base = "E:/experiments/e5/"

try:
    # check index ranks
    for nv in test_params:
        out_dir=f"{out_dir_base}{nv}/"
        os.makedirs(out_dir,exist_ok=True)
        images_list=images_list_base

        restrict_hypos = 5
        print(f"input width {nv}: verifying {restrict_hypos} hypotheses")

        # restrict_hypos=20
        if len(os.listdir(f"{out_dir_base}/{nv}")) < 1000:
            # run georeferencing
            cmd = f""" python main.py {images_list} {sheets} -r {restrict_hypos} --isize {nv}"""# --noimg
            os.system(cmd)
            for file in glob.glob(f"{out_dir_base}/tmp/*"):
                shutil.move(file,out_dir)
            os.rmdir(f"{out_dir_base}/tmp/")
        else:
            print("georeferencing has already been run for", nv)
            continue
    
    # run evaluation scripts
    for nv in test_params:
        # evaluate retrieval hit rate
        out_dir=f"{out_dir_base}{nv}/"
        os.makedirs(out_dir, exist_ok=True)
        
        if os.path.isfile(f"{out_dir}/eval_result.csv"):
            print("already evaluated retrieval")
            continue
       
        eval_logs(out_dir)
        shutil.move("eval_result.csv", f"{out_dir}/eval_result.csv")
    for nv in test_params:
        # evaluate georef accuracy
        out_dir=f"{out_dir_base}/{nv}/"
        
        if os.path.isfile(f"{out_dir}/baseline_georef_scores.csv"):
            print("already evaluated georef")
            continue

        calc_and_dump(sheets, out_dir)

    # compare different runs
    results = []
    for idx,nv in enumerate(test_params):        
        out_dir=f"{out_dir_base}{nv}/"
        r = get_all_results(out_dir, nv, key="mae m")
        r2 = get_all_results(out_dir, nv, key="mae px")
        r["Mittel Genauigkeit Pixel"] = r2["Mittel Genauigkeit"]
        r["Median Genauigkeit Pixel"] = r2["Median Genauigkeit"]
        results.append(r)
    
    r = get_all_results("E:/experiments/e2/", 1200, key="mae m")
    r2 = get_all_results("E:/experiments/e2/", 1200, key="mae px")
    r["Mittel Genauigkeit Pixel"] = r2["Mittel Genauigkeit"]
    r["Median Genauigkeit Pixel"] = r2["Median Genauigkeit"]
    results.append(r)
    # results.append(get_all_results("E:/experiments/e2/", 1200, key="mae m")) # add baseline as first result
    
    # make comparison figure
    make_figure(results, out_dir_base, "Bildgröße", max_error=500, x_type="string")#10)

finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")


# 200
# total mean error: 316.1678885584286 mae m
# median MAE: 285.5234974131027 mae m

# total mean error: 1.631699808909482 mae px
# median MAE: 1.4999423041515558 mae px

# 500
# total mean error: 114.44170488243113 mae m
# median MAE: 78.44914453058915 mae m

# total mean error: 1.429459716697751 mae px
# median MAE: 1.0258669189167278 mae px


# 1200
# total mean error: 42.33741066896848 mae m
# median MAE: 10.075360889521775 mae m

# total mean error: 1.364021995082409 mae px
# median MAE: 0.3139840225790542 mae px
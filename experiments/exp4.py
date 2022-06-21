# Experiment 4: Robustheit gegen Occlusion/Verdeckung/fehlende Signaturen

from math import ceil
import os, shutil

def make_figure(results, out_dir):
    from matplotlib import pyplot as plt
    xs = [x[0] for x in results]
    ret_scores = [x[1] for x in results]
    reg_mean_scores = [x[2] for x in results]
    reg_median_scores = [x[3] for x in results]

    ax = plt.gca()
    plt.xlabel("Occlusion [%]")
    ax2 = ax.twinx()
    ax.plot(ret_scores,label="Erfolgsquote")
    ax.set_ylabel('Erfolgsquote [%]')
    ax2.plot(reg_mean_scores,label="Mittel")
    ax2.plot(reg_median_scores,label="Median")
    ax2.set_ylabel('Fehler [m]')
    plt.legend()
    plt.savefig(out_dir+"comparison.png")

sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
images_list_base = "E:/data/deutsches_reich/osm_baseline/list.txt"

# set config
shutil.move("config.py", "config.py.old")
shutil.copy("experiments/config_e4.py", "config.py")

occlusion_params = [300,500,700]
data_dir_base = "E:/data/osm_baseline_degraded/"
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
            cmd = f"make_osm_baseline.py {sheets} {images_list_base} {data_dir} --saltpepper {nv}"
            os.system(cmd)

    for nv in occlusion_params:
        if os.path.isfile(f"{out_dir}/index_result.csv"):
            print("index already present")
            continue
        images_list=f"{data_dir_base}{nv}/list.txt"
        out_dir=f"{out_dir_base}{nv}/"
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

        restrict_hypos = ceil(restrict_hypos/5)*5 # round to next higher step of 5
        restrict_hypos = max(restrict_hypos,5) # leave some room for errors :)
        print(f"noise amount {nv}: verifying {restrict_hypos} hypotheses")

        # restrict_hypos=20
        # run georeferencing
        cmd = f""" python main.py {images_list} {sheets} -r {restrict_hypos}"""# --noimg
        os.system(cmd)
        shutil.move("E:/experiments/e3/tmp/",out_dir)
    
    exit()

    # run eval scripts
    for nv in occlusion_params:
        cmd = "python eval_logs.py"
        os.system(cmd)
        out_dir=f"{out_dir_base}{nv}/"
        os.makedirs(out_dir, exist_ok=True)
        shutil.move("eval_result.csv", f"{out_dir}/eval_result.csv")
    for nv in occlusion_params:
        # cmd = "python -m experiments.eval_baseline_georef"
        out_dir=f"{out_dir_base}{nv}/"
        from eval_baseline_georef import calc_and_dump
        calc_and_dump(sheets, out_dir)

    # compare different runs
    results = []
    # to do: add baseline as first result
    for idx,nv in enumerate(occlusion_params):
        out_dir=f"{out_dir_base}{nv}/"
        data_dir=f"{data_dir_base}{nv}/"
        # retrieval scores
        from experiments.summary_retrieval_logs import make_summary 
        with open(f"{out_dir}/retrieval_summary.txt","w") as outfile:
            percent_correct, mean_index, max_rank = make_summary(f"{out_dir}/eval_result.csv", outfile=outfile)

        # georef scores
        from experiments.summary_baseline_georef import load_results,filter_results,summary_and_fig
        results = load_results(out_dir+"baseline_georef_scores.csv")
        results = filter_results(results, f"{out_dir}/eval_result.csv")
        mean_mae, median_mae = summary_and_fig(results,out_dir)
        
        with open(f"{data_dir}/occlusion.txt") as fp:
            vals = []
            for line in fp:
                sheet,occl_val = line.strip().split(",")
                vals.append(float(occl_val))
        percent_occlusion = sum(vals)/len(vals)
        
        results.append([percent_occlusion, retrieval_score, mean_mae, median_mae])
    
    # make comparison figure
    make_figure(results, out_dir_base)

finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")
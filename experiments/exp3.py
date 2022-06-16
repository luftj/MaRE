# Experiment 3: Robustheit gegen Rauschen

from math import ceil
import os, shutil

def make_figure(results, out_dir):
    from matplotlib import pyplot as plt
    xs = [x[0] for x in results]
    ret_scores = [x[1] for x in results]
    reg_mean_scores = [x[2] for x in results]
    reg_median_scores = [x[3] for x in results]

    ax = plt.gca()
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
shutil.copy("experiments/config_e3.py", "config.py")

noise_values = [0.2,0.4,0.6]
noise_values = [0.6]
data_dir_base = "E:/data/osm_baseline_degraded/"
out_dir_base = "E:/experiments/e3/"

try:
    # make degraded dataset
    for nv in noise_values:
        data_dir=f"{data_dir_base}{nv}/"
        try:
            # check for data
            os.makedirs(data_dir, exist_ok=False)
        except OSError:
            print(f"data for {nv} already present")
        else:
            cmd = f"make_osm_baseline.py {sheets} {images_list_base} {data_dir} --saltpepper {nv}"
            os.system(cmd)

    for nv in noise_values:
        images_list=f"{data_dir_base}{nv}/list.txt"
        out_dir=f"{data_dir_base}{nv}/"

        # determine necessary number of spatial verifications
        # cmd = f"""python indexing.py --list {images_list} {sheets}"""
        # os.system(cmd)
        # shutil.move("index_result.csv", out_dir + "index_result.csv")
        from indexing import search_list
        lps = search_list(images_list)
        restrict_hypos = 0
        with open(f"{out_dir}/index_result.csv","w") as fp:
            for sheet,rank in lps:
                fp.write("%s : %d\n"%(sheet,rank))
                if rank > restrict_hypos:
                    restrict_hypos = rank
        
        restrict_hypos = ceil(max(restrict_hypos)/5)*5 # round to next higher step of 5
        print(f"noise amount {nv}: verifying {restrict_hypos} hypotheses")

        # restrict_hypos=20
        # run georeferencing
        cmd = f""" python main.py {images_list} {sheets} -r {restrict_hypos}"""# --noimg
        os.system(cmd)
        shutil.move("E:/experiments/e3/tmp/",out_dir)
    
    exit()

    # run eval scripts
    for nv in noise_values:
        cmd = "python eval_logs.py"
        os.system(cmd)
        out_dir=f"{out_dir_base}{nv}/"
        os.makedirs(out_dir, exist_ok=True)
        shutil.move("eval_result.csv", f"{out_dir}/eval_result.csv")
    for nv in noise_values:
        # cmd = "python -m experiments.eval_baseline_georef"
        out_dir=f"{out_dir_base}{nv}/"
        from eval_baseline_georef import calc_and_dump
        calc_and_dump(sheets, out_dir)

    # compare different runs
    results = []
    # to do: add baseline as first result
    for nv in noise_values:
        # retrieval scores
        from experiments.summary_retrieval_logs import make_summary 
        with open(f"{out_dir}/retrieval_summary.txt","w") as outfile:
            percent_correct, mean_index, max_rank = make_summary(f"{out_dir}/eval_result.csv", outfile=outfile)

        # georef scores
        from experiments.summary_baseline_georef import load_results,filter_results,summary_and_fig
        results = load_results(out_dir+"baseline_georef_scores.csv")
        results = filter_results(results, f"{out_dir}/eval_result.csv")
        mean_mae, median_mae = summary_and_fig(results,out_dir)
        
        results.append([nv, retrieval_score, mean_mae, median_mae])
    
    # make comparison figure
    make_figure(results, out_dir_base)

finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")
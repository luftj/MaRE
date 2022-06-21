# Experiment 3: Robustheit gegen Rauschen

from math import ceil
import os, shutil, glob
from eval_logs import eval_logs
from experiments.eval_baseline_georef import calc_and_dump
from experiments.summary_retrieval_logs import make_summary
from experiments.summary_baseline_georef import load_results,filter_results,summary_and_fig

def get_all_results(out_dir, amount):
    with open(f"{out_dir}/retrieval_summary.txt","w") as outfile:
            percent_correct = make_summary(f"{out_dir}/eval_result.csv", outfile=outfile)

    # georef scores
    georef_scores = load_results(out_dir+"baseline_georef_scores.csv")
    georef_scores = filter_results(georef_scores, f"{out_dir}/eval_result.csv")
    mean_mae, median_mae = summary_and_fig(georef_scores,out_dir)

    # index scores should come from index_result, in eval_result they are limited to restrict number
    ranks = []
    with open(f"{out_dir}/index_result.csv") as fp:
        for line in fp:
            sheet,rank = line.split(" : ")
            ranks.append(int(rank))
    mean_index = sum(ranks)/len(ranks)
    median_index= sorted(ranks)[len(ranks)//2]
    max_rank = max(ranks)
    return {
        "type": "Rauschen",
        "amount": amount, 
        "Erfolgsquote": percent_correct, 
        "Median Index": median_index,
        "Mittel Index": mean_index,
        "Max. Index": max_rank,
        "Mittel Genauigkeit": mean_mae, 
        "Median Genauigkeit": median_mae
    }

def make_figure(results, out_dir):
    from matplotlib import pyplot as plt
    print(results)
    name = results[0]["type"]
    xs = [int(x["amount"]*100) for x in results]
    ret_scores = [x["Erfolgsquote"]*100 for x in results]
    reg_mean_scores = [x["Mittel Genauigkeit"] for x in results]
    reg_median_scores = [x["Median Genauigkeit"] for x in results]

    plt.close()
    ax = plt.gca()
    plt.xlabel(f"Anteil {name} [%]")
    plt.xticks(ticks=range(len(xs)),labels=xs)
    ax2 = ax.twinx()
    ax.plot(ret_scores,label="Erfolgsquote",c="r")
    ax.set_ylabel('Erfolgsquote [%]',color="r")
    ax2.plot(reg_mean_scores,label="Mittel")
    ax2.plot(reg_median_scores,label="Median")
    ax2.set_ylabel('Fehler [m]')
    plt.legend()
    plt.savefig(f"{out_dir}/comparison_{name}.png")

sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
images_list_base = "E:/data/deutsches_reich/osm_baseline/list.txt"

# set config
shutil.move("config.py", "config.py.old")
shutil.copy("experiments/config_e3.py", "config.py")

noise_values = [0.2,0.4,0.6]
noise_values = [0.6,0.7,0.8,0.9]
noise_values = [0.6,0.9]
noise_values = [0.9]
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
        out_dir=f"{out_dir_base}{nv}/"
        if os.path.isfile(f"{out_dir}/index_result.csv"):
            print("index already present")
            continue
        os.makedirs(out_dir,exist_ok=True)

        from indexing import search_list
        lps = search_list(images_list)
        with open(f"{out_dir}/index_result.csv","w") as fp:
            for sheet,rank in lps:
                fp.write("%s : %d\n"%(sheet,rank))
        
    # determine necessary number of spatial verifications
    for nv in noise_values:
        images_list=f"{data_dir_base}{nv}/list396.txt"
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
        print(f"noise amount {nv}: verifying {restrict_hypos} hypotheses")

        # restrict_hypos=20
        # run georeferencing
        cmd = f""" python main.py {images_list} {sheets} -r {restrict_hypos}"""# --noimg
        os.system(cmd)
        shutil.move(glob.glob("E:/experiments/e3/tmp/*"),out_dir)
    
    # run evaluation scripts
    for nv in noise_values:
        # evaluate retrieval hit rate
        out_dir=f"{out_dir_base}{nv}/"
        os.makedirs(out_dir, exist_ok=True)
        
        if os.path.isfile(f"{out_dir}/eval_result.csv"):
            print("already evaluated retrieval")
            continue
       
        eval_logs(out_dir)
        shutil.move("eval_result.csv", f"{out_dir}/eval_result.csv")
    for nv in noise_values:
        # evaluate georef accuracy
        out_dir=f"{out_dir_base}{nv}/"
        
        if os.path.isfile(f"{out_dir}/baseline_georef_scores.csv"):
            print("already evaluated georef")
            continue

        calc_and_dump(sheets, out_dir)

    # compare different runs
    results = []
    results.append(get_all_results("E:/experiments/e2/", 0)) # add baseline as first result
    for nv in noise_values:
        out_dir=f"{out_dir_base}{nv}/"
        results.append(get_all_results(out_dir, nv))
    
    # make comparison figure
    make_figure(results, out_dir_base)

finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")
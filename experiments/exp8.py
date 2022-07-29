# experiment 8: case study KDR100

from asyncore import write
from math import ceil
import os, shutil, glob

from experiments.summary_retrieval_logs import make_summary

def make_register_figs(out_dir, f="eval_georef_result.csv", max_error_cap=400, n_bins=15, x_step=50, y_step=10):
    # file="E:/experiments/e2/0.7/baseline_georef_scores.csv"
    errors_raw = []
    # read results from file
    file = f"{out_dir}/{f}"
    with open(file) as fr:
        fr.readline() # header
        for line in fr:
            sheet, mae, rmse = line.split(";")
            errors_raw.append(float(mae))
    
    from matplotlib import pyplot as plt
    plt.close()
    plt.grid(axis="y",linestyle='dotted')
    
    greater = len([x  for x in errors_raw if x >= max_error_cap])
    print(f"{greater} above {max_error_cap} ({greater/len(errors_raw)*100}%)")
    errors = [x  for x in errors_raw if x < max_error_cap]
    n, b, p = plt.hist(errors, bins=n_bins, label="konvergiert")
    print(n)
    print(b)
    plt.bar(max_error_cap,greater, width=(b[-1]-b[-2]),color="red",label="schlecht/nicht konvergiert")
    plt.xticks(
        list(range(x_step,max_error_cap,x_step))+[max_error_cap],
        list(range(x_step,max_error_cap,x_step))+[f">{max_error_cap}"])
    plt.yticks(range(0,(int(max(n))//y_step+1)*y_step,y_step))
    plt.ylabel("Anzahl Blätter")
    plt.xlabel("Fehler [m]")
    mean_error = sum(errors_raw)/len(errors_raw)
    median_error = sorted(errors_raw)[len(errors_raw)//2]
    print(f"{len([x for x in errors_raw if x > mean_error])} above mean {mean_error}")
    if median_error < max_error_cap:
        plt.vlines(
            [median_error],
            ymin=0,ymax=max(n),
            colors=["orange"],
            linewidth=2,
            label="Median Fehler")
    plt.legend()

    plt.savefig(out_dir+"/kdr100_georef_hist.png")
    plt.show()

def get_all_results(out_dir):
    with open(f"{out_dir}/retrieval_summary.txt","w") as outfile:
            percent_correct = make_summary(f"{out_dir}/eval_result.csv", outfile=outfile)

    # index scores should come from index_result, in eval_result they are limited to restrict number
    ranks = []
    with open(f"{out_dir}/index_result.csv") as fp:
        for line in fp:
            sheet,rank = line.split(" : ")
            ranks.append(int(rank))
    mean_index = sum(ranks)/len(ranks)
    median_index= sorted(ranks)[len(ranks)//2]
    max_rank = max(ranks)
    return percent_correct, mean_index, median_index, max_rank

if __name__ == "__main__":
    sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
    images_list = "E:/data/deutsches_reich/SLUB/cut/raw/list_507.txt"
    annotations = "E:/data/deutsches_reich/SLUB/cut/raw/annotations.csv"

    # save old config
    shutil.move("config.py", "config.py.old")
    shutil.copy("experiments/config_e8.py", "config.py")

    from eval_logs import eval_logs
    from eval_georef import summary_and_fig

    data_dir = "E:/data/deutsches_reich/SLUB/cut/raw/"
    out_dir = "E:/experiments/e8/"
    os.makedirs(out_dir,exist_ok=True)

    try:
        # check for index
        os.makedirs(f"E:/experiments/idx_kdr100/index/keypoints",exist_ok=True)
        index_exists = len(os.listdir(f"E:/experiments/idx_kdr100/index/keypoints")) > 0
        print("index present:", index_exists, f"E:/experiments/idx_kdr100/index/keypoints")
        if not index_exists:
            # build index
            cmd = f"""python indexing.py --rebuild {sheets}"""
            os.system(cmd)

        # eval index ranks
        if os.path.isfile(f"{out_dir}/index_result.csv"):
            print("indexing reults already present")
        else:
            from indexing import search_list
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
        print(f"KDR100: verifying {restrict_hypos} hypotheses")

        # run georeferencing
        if len(glob.glob(f"{out_dir}/2022*.log")) < 1: # to do: check something smarter here
            cmd = f""" python main.py {images_list} {sheets} -r {restrict_hypos}"""# --noimg
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
            print("appending...")
            with open(f"{out_dir}/georef_summary.txt","w") as outfile:
                summary_and_fig(annotations, sheets, outfile=outfile, append_to=f"{out_dir}/eval_georef_result.csv")
        else:
            with open(f"{out_dir}/georef_summary.txt","w") as outfile:
                summary_and_fig(annotations, sheets, outfile=outfile)
            shutil.move("eval_georef_result.csv", f"{out_dir}/eval_georef_result.csv")
            shutil.move("georef_error.png", f"{out_dir}/georef_error.png")

        # make summary and figs
        percent_correct, mean_index, median_index, max_rank = get_all_results(out_dir)
        with open(f"{out_dir}/retrieval_summary.txt","a") as outfile:
            outfile.write(f"mean index: {mean_index}\n")
            outfile.write(f"median index: {median_index}\n")
            outfile.write(f"max index: {max_rank}\n")
        make_register_figs(out_dir)

    finally:
        # reset config
        os.remove("config.py")
        shutil.move("config.py.old", "config.py")
import os, shutil

from eval_scripts.plot_index import plot_single_distribution
from experiments.eval_baseline_georef import calc_and_dump
from experiments.summary_baseline_georef import load_results,filter_results,summary_and_fig

def make_register_hist(out_dir, f="eval_georef_result.csv"):
    # file="E:/experiments/e2/0.7/baseline_georef_scores.csv"
    errors_raw = []
    # read results from file
    file = f"{out_dir}/{f}"
    with open(file) as fr:
        fr.readline() # header
        for line in fr:
            sheet, mae_px, mae = line.split(";")
            errors_raw.append(float(mae))
    
    from matplotlib import pyplot as plt
    plt.close()
    print(errors_raw)
    plt.grid(axis="y",linestyle='dotted')
    max_error_cap = 400
    greater = len([x  for x in errors_raw if x >= max_error_cap])
    print(f"{greater} above {max_error_cap}")
    errors = [x  for x in errors_raw if x < max_error_cap]
    n, b, p = plt.hist(errors, bins=15,label="konvergiert")
    print(n)
    print(b)
    plt.bar(max_error_cap,greater, width=(b[-1]-b[-2]),color="red",label="schlecht/nicht konvergiert")
    plt.xticks(
        list(range(50,max_error_cap,50))+[max_error_cap],
        list(range(50,max_error_cap,50))+[f">{max_error_cap}"])
    # print(max(n),(int(max(n))//10+1)*10)
    plt.yticks(range(0,(int(max(n))//100+1)*100,100))
    plt.ylabel("Anzahl Blätter")
    plt.xlabel("Fehler [m]")
    mean_error = sum(errors_raw)/len(errors_raw)
    median_error = sorted(errors_raw)[len(errors_raw)//2]
    print(f"{len([x for x in errors_raw if x > mean_error])} above mean {mean_error}")
    plt.vlines(
        [median_error],
        ymin=0,ymax=max(n),
        colors=["orange"],
        linewidth=2,
        label="Median Fehler")
    plt.legend()
    plt.savefig(out_dir+"/georef_hist.png")
    plt.show()

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
    if not os.path.isfile(out_dir+"eval_result.csv"):
        restrict_hypos=0
        cmd = f""" python main.py {images_list} {sheets} -r {restrict_hypos}"""# --crop"""# --noimg
        os.system(cmd)
        shutil.move("eval_result.csv", f"{out_dir}/eval_result.csv")

    # run eval scripts
    if not os.path.isfile(out_dir+"baseline_georef_scores.csv"):
        calc_and_dump(sheets, out_dir)

    # make figures for georef scores
    results = load_results(out_dir+"baseline_georef_scores.csv")
    results = filter_results(results, "E:/experiments/e1/eval_result.csv")
    if not os.path.isfile(f"{out_dir}/registration_summary.txt"):
        with open(f"{out_dir}/registration_summary.txt","w") as outfile:
            mean_mae, median_mae, error_results = summary_and_fig(results, out_dir, outfile=outfile)

    make_register_hist(out_dir,f="baseline_georef_scores.csv")

finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")
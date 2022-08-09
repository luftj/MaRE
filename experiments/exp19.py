# experiment 19: compare with manual georef

import os, shutil

sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
annotations = "E:/data/deutsches_reich/SLUB/cut/raw/annotations.csv"

# save old config
shutil.move("config.py", "config.py.old")
# get new config
shutil.copy("experiments/config_e19.py", "config.py")

from experiments.exp8 import make_register_figs
from experiments.exp8 import get_all_results
from eval_logs import eval_logs
from eval_georef import summary_and_fig

data_dir = "experiments/output/"
out_dir = "E:/experiments/e19/"
os.makedirs(out_dir,exist_ok=True)

def register_compare(manual_georef_results,auto_georef_results="E:/experiments/e8/eval_georef_result.csv", max_error_cap=400, n_bins=5, x_step=50, y_step=10):
    errors_raw = []
    errors_auto = {}
    # read results from file
    with open(auto_georef_results) as fr:
        fr.readline() # header
        for line in fr:
            sheet, mae, rmse = line.split(";")
            # errors_auto[sheet] = float(mae)
            errors_raw.append(float(mae))

    errors_manual = []
    # read results from file
    with open(manual_georef_results) as fr:
        fr.readline() # header
        for line in fr:
            sheet, mae, rmse = line.split(";")
            # errors_raw.append(errors_auto[sheet])
            errors_manual.append(float(mae))

    print(f"auto mean: {sum(errors_raw)/len(errors_raw)}")
    print(f"auto median: {sorted(errors_raw)[len(errors_raw)//2]}")
    print(f"manual mean: {sum(errors_manual)/len(errors_manual)}")
    print(f"manual median: {sorted(errors_manual)[len(errors_manual)//2]}")
    
    from matplotlib import pyplot as plt
    plt.close()
    plt.grid(axis="y",linestyle='dotted')
    
    greater = len([x  for x in errors_raw if x >= max_error_cap])
    print(f"{greater} above {max_error_cap} ({greater/len(errors_raw)*100}%)")
    errors = [x  for x in errors_raw if x < max_error_cap]
    n, b, p = plt.hist(errors, bins=n_bins, label="konvergiert")
    n, b, p = plt.hist(errors_manual, bins=b, label="manuell")
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

    plt.savefig(out_dir+"/georef_compare.png")
    plt.show()

try:    
    # evaluate georef accuracy of manual georefs
    if os.path.isfile(f"{out_dir}/eval_georef_result.csv"):
        print("already evaluated georef")
    else:
        with open(f"{out_dir}/georef_summary.txt","w") as outfile:
            summary_and_fig(annotations, sheets, outfile=outfile, downscale_factor=4)
        shutil.move("georef_error.png", f"{out_dir}/georef_error.png")

    # compare auto georef and manual georef
    # todo
    # load both georef results
    # make figure with comparison
    register_compare(manual_georef_results=f"{out_dir}/eval_georef_result.csv")

finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")
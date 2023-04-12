# exp 15 konfidenz lokalisierung

# rerun without skips or early termination
from math import ceil
import os, shutil, glob

from experiments.summary_retrieval_logs import make_summary

from experiments.exp8 import make_register_figs, get_all_results

if __name__ == "__main__":
    sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
    images_list = "E:/data/deutsches_reich/SLUB/cut/raw/list.txt"
    annotations = "E:/data/deutsches_reich/SLUB/cut/raw/annotations.csv"

    # save old config
    shutil.move("config.py", "config.py.old")
    shutil.copy("experiments/config_e15.py", "config.py")

    from eval_logs import eval_logs
    from eval_georef import summary_and_fig

    data_dir = "E:/data/deutsches_reich/SLUB/cut/raw/"
    out_dir = "E:/experiments/e15/"
    os.makedirs(out_dir,exist_ok=True)

    try:            
        # determine necessary number of spatial verifications
        restrict_hypos = 0
        with open(f"{out_dir}/../e8/index_result.csv") as fp:
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
            shutil.move("georef_error.png", f"{out_dir}/georef_error.png")

        # # make summary and figs
        # percent_correct, mean_index, median_index, max_rank = get_all_results(out_dir)
        # with open(f"{out_dir}/retrieval_summary.txt","a") as outfile:
        #     outfile.write(f"mean index: {mean_index}\n")
        #     outfile.write(f"median index: {median_index}\n")
        #     outfile.write(f"max index: {max_rank}\n")
        # make_register_figs(out_dir)

    finally:
        # reset config
        os.remove("config.py")
        shutil.move("config.py.old", "config.py")

# load retrieval results
import csv
results = {}
with open(f"{out_dir}/eval_result.csv") as fr:
    reader = csv.DictReader(fr, delimiter=';', quotechar='"')
    for row in reader:
        sheet = row["ground truth"]
        mahalanobis = float(row[" mahalanobis"])
        lowes = float(row[" Lowe's test ratio"])
        correct = (int(row[" ground truth position"]) == 0)
        results[sheet] = {
            "mahalanobis": mahalanobis,
            "lowes": lowes,
            "correct": correct
        }

from matplotlib import pyplot as plt

maha = [x["mahalanobis"] for x in results.values() if x["correct"]]
lowes = [x["lowes"] for x in results.values() if x["correct"]]
maha_wrong  = [x["mahalanobis"] for x in results.values() if not x["correct"]]
lowes_wrong  = [x["lowes"] for x in results.values() if not x["correct"]]
print("num below 8 sigma",len([x for x in maha if x <=8]))

min_maha = max(maha_wrong)
print(min_maha)
print(f"num correct below {min_maha} (max wrong maha)",len([x for x in maha if x <=min_maha]))

plt.close()
plt.scatter(maha,lowes, label="Korrekte Vorhersage")
plt.scatter(maha_wrong,lowes_wrong, color="r", marker="+", label="Falsche Vorhersage")
plt.axhline(0.7, c="y", linestyle="-.", label="Lowe's Grenzwert 0.7")
plt.axvline(min_maha, c="g", linestyle="--", label="Mahalanobis-Distanz %.2f σ"%min_maha)
plt.ylabel("Lowe's Test")
plt.xlabel("Mahalanobis-Distanz")
plt.legend()
out_dir = "./docs/eval_diagrams/"

filetype="pdf"
dpi_text = 1/72
fig_width=420
fig_height=250
dpi = 600
params = {'backend': filetype,
        'axes.labelsize': 11,
        'font.size': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        # 'figure.figsize': [7, 4.5],
        'text.usetex': True,
        'figure.figsize': [fig_width*dpi_text,fig_height*dpi_text]
        }
plt.rcParams.update(params)
plt.savefig(f"{out_dir}/maha_lowe_scatter."+filetype,dpi=dpi, bbox_inches = 'tight')
plt.show()


# plot FP/TP for different score values

# metriken:
# mahalonobis


plt.rcParams.update({'figure.figsize': [fig_width/2*dpi_text,fig_height*dpi_text]})
import numpy as np
fns = []
fps = []
pois = []
for i in np.arange(0,20,0.25):
    fn_rate = len([x for x in maha if x < i])#/len(maha+maha_wrong)
    fp_rate = len([x for x in maha_wrong if x > i])#/len(maha+maha_wrong)
    fns.append(fn_rate)
    fps.append(fp_rate)
    if i in [5,6,7,8]:
        pois.append((int(i),fp_rate,fn_rate))

fn_rate = len([x for x in maha if x < min_maha])#/len(maha+maha_wrong)
fp_rate = len([x for x in maha_wrong if x > min_maha])#/len(maha+maha_wrong)
pois.append((min_maha,fp_rate,fn_rate))

plt.title("ROC Mahalanobis")
plt.plot(fps,fns)
for t,x,y in pois:
    plt.annotate(t,(x,y))
plt.ylabel('Falsch negativ')# Rate
plt.xlabel('Falsch positiv')# Rate
# plt.xticks(np.arange(0.0,0.2,0.1))
# plt.yticks(np.arange(0.0,1.0,0.1))

# plt.rcParams.update(params)
plt.savefig(f"{out_dir}/roc_maha."+filetype,dpi=dpi, bbox_inches = 'tight')
plt.show()

# lowes test

fns = []
fps = []
pois = []
for i in np.arange(0,1,0.01):
    fn_rate = len([x for x in lowes if x > i])#/len(lowes+lowes_wrong)
    fp_rate = len([x for x in lowes_wrong if x < i])#/len(lowes+lowes_wrong)
    fns.append(fn_rate)
    fps.append(fp_rate)
    if i in [0.5,0.6,0.7,0.8,0.9]:
        pois.append((i,fp_rate,fn_rate))

fn_rate = len([x for x in lowes if x > 0.7])#/len(lowes+lowes_wrong)
fp_rate = len([x for x in lowes_wrong if x < 0.7])#/len(lowes+lowes_wrong)
pois.append((0.7,fp_rate,fn_rate))

plt.title("ROC Lowe's")
plt.plot(fps,fns)
for t,x,y in pois:
    plt.annotate(t,(x,y))
plt.ylabel('Falsch negativ')# Rate
plt.xlabel('Falsch positiv')# Rate
# plt.xticks(np.arange(0.0,0.2,0.1))
# plt.yticks(np.arange(0.0,1.0,0.1))

# plt.rcParams.update(params)
# plt.rcParams.update({'figure.figsize': [fig_width/2*dpi_text,fig_height*dpi_text]})
plt.savefig(f"{out_dir}/roc_lowe."+filetype,dpi=dpi, bbox_inches = 'tight')
plt.show()

# codebook response threshold not possible to extract like this
# exp 20: registration confidence

import os
import csv
from matplotlib import pyplot as plt

results_path = "eval_result.csv"
results_path = "E:/experiments/e15/eval_result.csv"
georefs_path = "E:/experiments/e15/"
georef_results_path = "E:/experiments/e15/eval_georef_result.csv"
out_dir = "E:/experiments/e20/"
os.makedirs(out_dir, exist_ok=True)

ret_results = {}
with open(results_path) as fr:
    reader = csv.DictReader(fr, delimiter=';', quotechar='"')
    for row in reader:
        sheet, ecc = row["ground truth"],float(row[" ecc score"])
        if int(row[" ground truth position"]) == 0:
            ret_results[sheet] = { 
                "ecc":ecc,
                "time":float(row[" registration time"])
            }

scores = [x["ecc"] for x in ret_results.values()]
mean_ecc = sum(scores)/len(scores)
print("mean ecc over sheets", mean_ecc)
print("median ecc over sheets", sorted(scores)[len(scores)//2])
print("num of correctly retrieved sheets", len(scores))

reg_results = {}
with open(georef_results_path) as fr:
    fr.readline()
    for line in fr:
        sheet, mae, rmse = line.strip().split("; ")
        # sheet, mae_px, mae_m = line.strip().split(";") # baseline
        # mae = mae_m # baseline
        reg_results[sheet] = float(mae)

eccs = [ret_results[s]["ecc"] for s in reg_results.keys()]
maes = reg_results.values()
print("number of registered sheets:",len(maes))
mean_mae = sum(maes)/len(maes)
print("mean mae over registered sheets", mean_mae)
print("median mae over registered  sheets", sorted(maes)[len(maes)//2])

thresh = 0.76
over_thresh = [x for x in eccs if x > thresh]
mae_over_thresh = [mae for s,mae in reg_results.items() if ret_results[s]["ecc"] > thresh]
print(f"num over {thresh}",len(over_thresh))
print(f"mean mae over thresh:",sum(mae_over_thresh)/len(mae_over_thresh))

times = [ret_results[s]["time"] for s in reg_results.keys()]

plt.scatter(eccs,maes)
plt.xlabel("ECC")
plt.ylabel("MAE [m]")
plt.savefig(out_dir+"ecc_mae.png")
plt.show()

plt.scatter(times,maes)
plt.xlabel("time [s]")
plt.ylabel("MAE [m]")
plt.savefig(out_dir+"t_mae.png")
plt.show()

ecc_mae = [(ret_results[s]["ecc"],mae) for s,mae in reg_results.items()]

import numpy as np
fns = []
fps = []
tps = []
pois = []
for i in np.arange(0,1,0.05):
    fn_rate = len([x for x in ecc_mae if x[0] < i and x[1] < 400])#/len(maha+maha_wrong)
    fp_rate = len([x for x in ecc_mae if x[0] > i and x[1] > 400])#/len(maha+maha_wrong)
    tp_rate = len([x for x in ecc_mae if x[0] > i and x[1] < 400])
    tn_rate = len([x for x in ecc_mae if x[0] < i and x[1] > 400])
    fns.append(fn_rate)
    fps.append(fp_rate)
    if i in [0.2,0.4,0.5,0.6,0.7,0.8]:
        pois.append((("%.1f"%i),fp_rate,fn_rate))


plt.title("ROC ECC")
plt.plot(fps,fns)
for t,x,y in pois:
    plt.annotate(t,(x,y))
plt.ylabel('False Negative')# Rate
plt.xlabel('False Positive')# Rate
# plt.xticks(np.arange(0.0,0.2,0.1))
# plt.yticks(np.arange(0.0,1.0,0.1))
plt.savefig(f"{out_dir}/roc_ecc.png")
plt.show()

from glob import glob

transforms = {}

for file in glob(f"{georefs_path}/transform*"):
    sheet = file.split("_")[-1].split(".")[0]
    transform = np.load(file)
    # print(transform)
    # print(np.ndarray.flatten(transform,order="F"))
    # exit()
    # transforms[sheet] = np.ndarray.flatten(transform)
    transforms[sheet] = np.ndarray.flatten(transform,order="F")[0:4]
    # transforms[sheet] = np.ndarray.flatten(transform,order="F")

vals = np.stack(transforms.values())
print(vals)
mean = np.mean(vals, axis=0)
std = np.std(vals,axis=0)
print("mean transform",mean)
print("std transform",std)

outliers = []
inliers = []
for s,t in transforms.items():
    if np.any(np.abs(mean-t) > 1 * std):
        outliers.append(s)
    else:
        inliers.append(s)

print("outliers",len(outliers))
print("inliers",len(inliers))

# devs = {s:(abs(t[0])+abs(t[3])+abs(t[4]))/3 for s,t in transforms.items()}
# thresh=0.00297
# print(f"<{thresh}",len([x for x in devs.values() if x < thresh]))
# plt.close()
# plt.scatter(devs.values(),[reg_results[s] for s in devs.keys()])
# plt.vlines([thresh],0,20000,colors=["r"])
# plt.hlines([400],0,max(devs.values()),colors=["y"])
# plt.xlabel("size params")
# plt.ylabel("MAE [m]")
# plt.show()
# exit()


devs = {s:max(np.abs(mean-t)/std) for s,t in transforms.items()}
thresh=0.428
print(f"<{thresh}",len([x for x in devs.values() if x < thresh]))
plt.close()
plt.scatter(devs.values(),[reg_results[s] for s in devs.keys()])
plt.vlines([thresh],0,20000,colors=["r"])
plt.hlines([400],0,max(devs.values()),colors=["y"])
plt.xlabel("max distance to mean")
plt.ylabel("MAE [m]")
for s in transforms.keys():
    x = max(np.abs(mean-transforms[s])/std)
    y = reg_results[s]
    plt.annotate(s,(x,y))
plt.savefig(f"{out_dir}/transform_maxdist.png")
plt.show()

devs = {s:np.mean(np.abs(mean-t)/std) for s,t in transforms.items()}
thresh=0.161
print(f"<{thresh}",len([x for x in devs.values() if x < thresh]))
plt.close()
plt.scatter(devs.values(),[reg_results[s] for s in devs.keys()])
plt.vlines([thresh],0,20000,colors=["r"])
plt.hlines([400],0,max(devs.values()),colors=["y"])
plt.xlabel("mean distance to mean")
plt.ylabel("MAE [m]")
plt.show()

devs = {s:sum(np.abs(mean-t)/std) for s,t in transforms.items()}
thresh=0.973
print(f"<{thresh}",len([x for x in devs.values() if x < thresh]))
plt.close()
plt.scatter(devs.values(),[reg_results[s] for s in devs.keys()])
plt.vlines([thresh],0,20000,colors=["r"])
plt.hlines([400],0,max(devs.values()),colors=["y"])
plt.xlabel("sum distance to mean")
plt.ylabel("MAE [m]")
plt.show()

devs = {s:((np.abs(mean-t)/std)[1]+(np.abs(mean-t)/std)[2])/2 for s,t in transforms.items()}
thresh=0.066
print(f"<{thresh}",len([x for x in devs.values() if x < thresh]))
plt.close()
plt.scatter(devs.values(),[reg_results[s] for s in devs.keys()])
plt.vlines([thresh],0,20000,colors=["r"])
plt.hlines([400],0,max(devs.values()),colors=["y"])
plt.xlabel("[1]+[2] distance to mean")
plt.ylabel("MAE [m]")
plt.show()

devs = {s:((np.abs(mean-t)/std)[0]+(np.abs(mean-t)/std)[3])/2 for s,t in transforms.items()}
thresh=0.122
print(f"<{thresh}",len([x for x in devs.values() if x < thresh]))
plt.close()
plt.scatter(devs.values(),[reg_results[s] for s in devs.keys()])
plt.vlines([thresh],0,20000,colors=["r"])
plt.hlines([400],0,max(devs.values()),colors=["y"])
plt.xlabel("[0]+[3] distance to mean")
plt.ylabel("MAE [m]")
plt.show()

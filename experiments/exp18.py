# expermiment 18 runtime registrierung
import csv
import ast

results_path = "eval_result.csv"
# results_path = "E:/experiments/e2/eval_result.csv"
results_path = "E:/experiments/e8/eval_result.csv"

all_times = []
times = []
num_kps = []
all_scores = []
avg_scores = []
perc_fg = []
ecc_scores = []
times_dict = {}
with open(results_path) as fr:
    reader = csv.DictReader(fr, delimiter=';', quotechar='"')
    for row in reader:
        sheet, time = row["ground truth"],float(row[" registration time"])
        time_arr = ast.literal_eval(row[" times"].strip())
        # time = sum(time_arr)/len(time_arr)
        # if True:
        if time != -1:
            times.append(float(time))
            num_kps.append(int(row[" number of detected keypoints"]))
            scores = ast.literal_eval(row[" scores"].strip())
            all_scores.append(scores)
            avg_scores.append(sum(scores)/len(scores))
            all_times.append(time_arr)
            perc_fg.append(float(row[" percent segmented"].replace(",",".")))
            ecc_scores.append(float(row[" ecc score"]))
            times_dict[sheet] = time

mean_time = sum(times)/len(times)
print("mean time per sheet", mean_time)
print("num checked maps", len(times))

from matplotlib import pyplot as plt
# plt.scatter(ecc_scores,times)
# -> no correlation

# plt.scatter(all_scores,all_times)
# plt.xlabel("hypothesis score")
# plt.ylabel("registration time")

plt.hist(times)
plt.xlabel("registration time")
plt.show()
# -> most are <1s. so probalby higher ecc iterations doesn't really help

# to do: correlate with georef scores?

maes = []
sheets = []

georef_results_path="eval_georef_result.csv"
georef_results_path = "E:/experiments/e8/eval_georef_result.csv"
# results_path = "/e/experiments/e2/baseline_georef_scores.csv"
with open(georef_results_path) as fr:
    fr.readline()
    for line in fr:
        sheet, mae, rmse = line.strip().split("; ")
        sheet, mae_px, mae_m = line.strip().split(";")
        mae = mae_m
        maes.append(float(mae))
        sheets.append(sheet)

print("mean mae",sum(maes)/len(maes))

plt.scatter(maes,[times_dict[x] for x in sheets])
plt.show()
# -> keine korrelation. nicht nur schlechte Lösungen brauchen lange


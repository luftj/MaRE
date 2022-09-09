# expermiment 17 runtime lokalisierung
import csv
import ast

results_path = "eval_result.csv"
results_path = "E:/experiments/e17b/eval_result.csv"
results_path = "E:/experiments/e8/eval_result.csv"
results_path = "E:/experiments/e15/eval_result.csv"

all_times = []
times = []
num_kps = []
all_scores = []
avg_scores = []
perc_fg = []
early_terminations = 0
localisation_failures = 0
with open(results_path) as fr:
    reader = csv.DictReader(fr, delimiter=';', quotechar='"')
    for row in reader:
        sheet, time = row["ground truth"],float(row[" avg time per sheet"])
        if time != -1:
            times.append(float(time))
            num_kps.append(int(row[" number of detected keypoints"]))
            scores = ast.literal_eval(row[" scores"].strip())
            all_times.append(ast.literal_eval(row[" times"].strip()))
            all_scores.append(scores)
            avg_scores.append(sum(scores)/len(scores))
            perc_fg.append(float(row[" percent segmented"].replace(",",".")))
            if len(scores) < 30:
                early_terminations += 1
            if int(row[" ground truth position"]) != 0:
                localisation_failures += 1

mean_time = sum(times)/len(times)
print("mean time per sheet", mean_time)
print("num checked maps", len(times))
print("num early termiantions", early_terminations)
print("failed localisations",localisation_failures)

from matplotlib import pyplot as plt
# plt.scatter(avg_scores,times)
# plt.scatter(perc_fg,times)
# plt.scatter(num_kps,times)
# -> no correlation between sheet stats and times

plt.scatter(all_scores,all_times)
plt.xlabel("hypothesis score")
plt.ylabel("validation time")
# -> antiproportional correlation in hypothesis validation?
# -> sometimes more ransac iterations with fewer keypoint-matches?
plt.show()

sv_runs = {
    1: "E:/experiments/e17a/eval_result.csv",
    10: "E:/experiments/e17c/eval_result.csv",
    20: "E:/experiments/e17d/eval_result.csv",
    30: "E:/experiments/e15/eval_result.csv"
}

sv_results = {}
for sv_its, results_path in sv_runs.items():
    num_correct = 0
    num_total = 0
    avg_time = 0
    with open(results_path) as fr:
        reader = csv.DictReader(fr, delimiter=';', quotechar='"')
        for row in reader:
            sheet, time = row["ground truth"],float(row[" avg time per sheet"])
            all_times = (ast.literal_eval(row[" times"].strip()))
            print(len(all_times))
            if int(row[" ground truth position"]) == 0:
                num_correct += 1
            num_total += 1
            avg_time += sum(all_times)
    avg_time /= num_total * sv_its
    sv_results[sv_its] = {
        "ratio_correct": num_correct/num_total,
        "avg_time": avg_time
    }

plt.plot([x["ratio_correct"] for x in sv_results.values()], label="ratio correct")
plt.plot([x["avg_time"] for x in sv_results.values()], label="avg time per sheet")
print(sv_results.keys())
plt.xticks(ticks=range(len(list(sv_results.keys()))),labels=list(sv_results.keys()))
plt.legend()
plt.show()
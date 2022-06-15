from csv import DictReader

# load csv
out_dir = "E:/experiments/e2/"
scores_file = out_dir+"baseline_georef_scores.csv"
 
results = []
with open(scores_file,'r') as file:
    reader = DictReader(file,delimiter=";")
    print("Data:")
    # printing each row of table as dictionary 
    for idx,row in enumerate(reader):
        # print(row)
        # row["sheet"] = idx
        results.append(dict(row))

print(results[:2])
# filter incorrect retrieval prediction
retrievel_result_file = "E:/experiments/e1/eval_result.csv"
wrong_preds = []
no_reg_solution = []
with open(retrievel_result_file, "r") as fr:
    reader = DictReader(fr,delimiter=";")
    for row in reader:
        if int(row[" ground truth position"]) > 0:
            wrong_preds.append(row["ground truth"])
        if float(row[" registration time"]) < 0:
            no_reg_solution.append(row["ground truth"])

print(wrong_preds)
prefilter_len = len(results)
results = list(filter(lambda x: x["sheet"] not in wrong_preds, results))
print(f"filtered {prefilter_len-len(results)} incorrect predictions")
print(no_reg_solution)
prefilter_len = len(results)
results = list(filter(lambda x: x["sheet"] not in no_reg_solution, results))
print(f"filtered {prefilter_len-len(results)} no registration solution")

# summary
print(f"{len(results)} sheets analysed")
error_results = [float(v["mae m"]) for v in results]
total_mean_mae = sum(error_results)/len(error_results)
print("total mean error: %f m" % total_mean_mae)

sheet_names = [k["sheet"] for k in results]
results_sorted = sorted(zip(sheet_names,error_results), key=lambda tup: tup[1])
sheet_names_sorted = [x[0] for x in results_sorted]
error_sorted = [x[1] for x in results_sorted]

median_error_mae = error_sorted[len(error_sorted)//2]
print("median MAE: %f m" % median_error_mae)

print("best sheets:",results_sorted[0:5])
print("worst sheets:",results_sorted[-5:])

print("sheets < 200m:",len([x for x in error_sorted if x < 200]))
print("sheets > 500m:",len([x for x in error_sorted if x > 500]))
print("sheets > mean:",len([x for x in error_sorted if x > total_mean_mae]))

# # filter absurd outliers
# high_errors = []
# for r in results:
#     if float(r["mae m"]) > 1000:
#         r["mae m"] = 1000
#         high_errors.append(r["sheet"])
# print(high_errors)
# print(f"{len(high_errors)} with error > 1000")
# error_results = [float(v["mae m"]) for v in results]
# sheet_names = [k["sheet"] for k in results]
# results_sorted = sorted(zip(sheet_names,error_results), key=lambda tup: tup[1])
# sheet_names_sorted = [x[0] for x in results_sorted]
# error_sorted = [x[1] for x in results_sorted]

# make figure
from matplotlib import pyplot as plt
plt.subplot(2, 1, 1)
plt.yscale("log")
plt.bar(sheet_names_sorted, error_sorted)
plt.axhline(total_mean_mae, c="g", linestyle="--", label="mean")
plt.annotate("%.0f" % total_mean_mae,(0,total_mean_mae + 100))
plt.axhline(median_error_mae, c="r", label="median")
plt.annotate("%.0f" % median_error_mae,(0,median_error_mae + 100))
plt.legend()
plt.title("average error per sheet [m]")
plt.subplot(2, 1, 2)
plt.xscale("log")
plt.title('error distribution total [m]')
plt.boxplot([error_sorted], vert=False, showmeans=True, labels=["mae m"], medianprops={"color":"r"})
plt.axhline(total_mean_mae, xmax=0, c="g", label="mean")
plt.axhline(median_error_mae, xmax=0, c="r", label="median")
plt.legend()
# plt.show()
plt.savefig(out_dir+"baseline_georef_error.png")

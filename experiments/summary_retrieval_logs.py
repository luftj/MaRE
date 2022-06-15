from csv import DictReader

retrievel_result_file = "E:/experiments/e1/eval_result.csv"
wrong_preds = []
no_reg_solution = []
results = []
with open(retrievel_result_file, "r") as fr:
    reader = DictReader(fr,delimiter=";")
    for row in reader:
        if int(row[" ground truth position"]) > 0:
            wrong_preds.append(row["ground truth"])
        if float(row[" registration time"]) < 0:
            no_reg_solution.append(row["ground truth"])
        results.append(dict(row))

print("num sheets:", len(results))
print("num wrong preds:", len(wrong_preds))
print("% coorect preds:", (len(results)-len(wrong_preds))/len(results))
idx_0 = [x for x in results if int(x[" index rank"]) == 0]
print("num with index rank 0:", len(idx_0))
print("% with index rank 0:", (len(idx_0))/len(results))
max_rank = max([int(x[" index rank"]) for x in results])
print("worst index rank:", max_rank, "sheet(s)", [x["ground truth"] for x in results if int(x[" index rank"])==max_rank])
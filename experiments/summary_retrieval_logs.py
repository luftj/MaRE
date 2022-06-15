from csv import DictReader

retrievel_result_file = "E:/experiments/e1/eval_result.csv"
wrong_preds = []
results = []
with open(retrievel_result_file, "r") as fr:
    reader = DictReader(fr,delimiter=";")
    for row in reader:
        if int(row[" ground truth position"]) > 0:
            wrong_preds.append(row["ground truth"])
        results.append(dict(row))

print("num sheets:", len(results))
print("num wrong preds:", len(wrong_preds))
print("% coorect preds:", (len(results)-len(wrong_preds))/len(results))
idx_0 = [x for x in results if int(x[" index rank"]) == 0]
print("num with index rank 0:", len(idx_0))
print("% with index rank 0:", (len(idx_0))/len(results))
index_ranks=[int(x[" index rank"]) for x in results]
max_rank = max(index_ranks)
print("worst index rank:", max_rank, "sheet(s)", [x["ground truth"] for x in results if int(x[" index rank"])==max_rank])

print("mean index rank:",sum(index_ranks)/len(index_ranks))
print("median index rank:",sorted(index_ranks)[len(index_ranks)//2])
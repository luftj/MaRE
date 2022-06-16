from csv import DictReader
import sys

retrievel_result_file = "E:/experiments/e1/eval_result.csv"

def make_summary(retrievel_result_file, outfile=sys.stdout):
    wrong_preds = []
    results = []
    with open(retrievel_result_file, "r") as fr:
        reader = DictReader(fr,delimiter=";")
        for row in reader:
            if int(row[" ground truth position"]) > 0:
                wrong_preds.append(row["ground truth"])
            results.append(dict(row))

    print("num sheets:", len(results), file=outfile)
    print("num wrong preds:", len(wrong_preds), file=outfile)
    percent_correct = (len(results)-len(wrong_preds))/len(results)
    print("% coorect preds:", percent_correct, file=outfile)
    idx_0 = [x for x in results if int(x[" index rank"]) == 0]
    print("num with index rank 0:", len(idx_0), file=outfile)
    print("% with index rank 0:", (len(idx_0))/len(results), file=outfile)
    index_ranks=[int(x[" index rank"]) for x in results]
    max_rank = max(index_ranks)
    print("worst index rank:", max_rank, "sheet(s)", [x["ground truth"] for x in results if int(x[" index rank"])==max_rank], file=outfile)

    mean_index = sum(index_ranks)/len(index_ranks)
    print("mean index rank:", mean_index, file=outfile)
    print("median index rank:",sorted(index_ranks)[len(index_ranks)//2], file=outfile)
    return percent_correct, mean_index, max_rank
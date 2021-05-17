import os
import ast
import csv
from matplotlib import pyplot as plt
import numpy as np
import glob
import cv2
import json
import operator
import argparse

def get_retrieval_results(results_file):
    prediction_results = []
    with open(results_file) as fr:
        for line in csv.DictReader(fr, delimiter=";", skipinitialspace=True):
            prediction_results.append(dict(line))
    return prediction_results

def maha_lowe(res, sheets_correct, sheets_incorrect, output_folder):
    # res = [x for x in res if float(x["mahalanobis"]) >= 0]
    maha_correct    = [float(x["mahalanobis"]) for x in res if x["ground truth"] in sheets_correct]
    maha_incorrect  = [float(x["mahalanobis"]) for x in res if x["ground truth"] in sheets_incorrect]
    lowes_correct   = [float(x["Lowe's test ratio"]) for x in res if x["ground truth"] in sheets_correct]
    lowes_incorrect = [float(x["Lowe's test ratio"]) for x in res if x["ground truth"] in sheets_incorrect]

    lowes_correct = [1 if x == -1 else x for x in lowes_correct]
    lowes_incorrect = [0 if x == -1 else x for x in lowes_incorrect]
    # todo: deal with predictions with only one RANSAC result, which subsequently have neither Maha nor Lowe
    print("num correct below 4 sigma: %d/%d" % (len([x for x in maha_correct if 0< x <=4]),len([x for x in maha_correct if x>0])))
    print("num incorrect below 4 sigma: %d/%d" % (len([x for x in maha_incorrect if x <=4]),len(maha_incorrect)))

    min_maha = int(max(maha_incorrect))
    print("min maha to split incorrect: %f" % min_maha)
    print("false rejections with this split: %d/%d" % (len(list(filter(lambda x: x<min_maha,maha_correct))),len(res)))

    plt.scatter(maha_correct,lowes_correct, label="correct")
    plt.scatter(maha_incorrect,lowes_incorrect, color="r", marker="+", label="incorrect")
    plt.axhline(0.7, c="y", linestyle="-.", label="0.7 test ratio")
    plt.axvline(min_maha, c="g", linestyle="--", label="%dσ"%min_maha)
    plt.ylabel("Lowe's test ratio")
    plt.xlabel("Mahalanobis distance")
    plt.legend()
    plt.title("Prediction correctness thresholds")
    plt.savefig(output_folder+"/maha_lowe.png")
    plt.close()

def rank_compare(res, sheets_correct, sheets_incorrect, output_folder):
    # index ranks of correct predictions
    ranks_correct = [int(x["index rank"]) for x in res if x["ground truth"] in sheets_correct]

    # index ranks of incorrect predictions
    ranks_incorrect = [int(x["index rank"]) for x in res if x["ground truth"] in sheets_incorrect]
    
    # plot both over each other in bar plot
    counts_correct = np.bincount(ranks_correct)
    counts_incorrect = np.bincount(ranks_incorrect)
    plt.bar(range(len(counts_correct)),counts_correct, width=1, label="correct")
    plt.bar(range(len(counts_incorrect)),counts_incorrect, width=1, label="incorrect")
    plt.yticks(range(max(counts_correct)+1))
    plt.ylabel("# sheets")
    plt.xlabel("index rank")
    plt.legend()
    plt.title("histogram of index ranks of %d predictions" % (len(ranks_correct)+len(ranks_incorrect)))
    plt.savefig(output_folder+"/rank_hist.png")
    plt.close()

def compare_special_cases(res, case, list_path):
    cases = {}
    with open(list_path) as fr:
        for line in fr:
            line=line.strip()
            sheet, edition = line.split(",")
            cases[sheet] = edition
    possible_cases = list(set(cases.values()))

    fig, (ax_acc, ax_score) = plt.subplots(1, 2)

    # accuracy
    accuracy_total = [x["ground truth position"] for x in res].count("0") / len(res) * 100
    labels = ["total"]
    bars = [accuracy_total]
    # ransac score
    score_total = sorted([max(ast.literal_eval(x["scores"])) for x in res if x["ground truth position"]=="0"], reverse=True)
    avg_total = sum(score_total)/len(score_total)
    for option in possible_cases:
        res_specific = [x for x in res if cases[x["ground truth"]]==option]
        # accuracy
        accuracy_specific = [x["ground truth position"] for x in res_specific].count("0") / len(res_specific) * 100
        labels.append(option)
        bars.append(accuracy_specific)
        # avg ransac score
        score_specific = sorted([max(ast.literal_eval(x["scores"])) for x in res_specific if x["ground truth position"]=="0"], reverse=True) # filter out incorrect
        avgscore_specific = sum(score_specific)/len(score_specific)
        line, = ax_score.plot(score_specific, label=option)
        ax_score.hlines(avgscore_specific,0,len(score_total),label="%s avg."%option, linestyle="--", color=line.get_color())
        
    # accuracy
    ax_acc.bar(labels,bars)
    ax_acc.set(ylabel="correct sheets [%]")
    ax_acc.set_title("accuracy")

    # avg ransac score
    line, = ax_score.plot(score_total, label="total")
    ax_score.hlines(avg_total,0,len(score_total),label="total avg.", linestyle="--", color=line.get_color())
    ax_score.set(ylabel="RANSAC score",xlabel="sheet")
    ax_score.legend()
    ax_score.set_title("score")
    fig.suptitle("comparing %s" % case)
    plt.savefig(output_folder+"/compare_%s.png" % case)
    plt.close()

def compare_scores(res, output_folder):
    # show best score with actual truth score
    sorted_res = sorted(res,key=lambda x: max(ast.literal_eval(x["scores"])), reverse=True)
    best_score = [max(ast.literal_eval(x["scores"])) for x in sorted_res]
    scores = [ast.literal_eval(x["scores"]) for x in sorted_res]
    index_ranks = [int(x["index rank"]) for x in sorted_res]
    # print(list(zip(scores,index_ranks)))
    true_score = [sc[ir] if (len(sc) > ir) else 0 for sc,ir in zip(scores,index_ranks)]
    # true_score = [score[index_ranks[idx]] if (index_ranks[idx]>=0 and len(score)) else 0 for idx,score in enumerate(scores)]
    # true_score = [ast.literal_eval(x["scores"])[int(x["index rank"])] if len(ast.literal_eval(x["scores"]))>0 else 0 for x in sorted_res]
    # compare with index rank
    index_rank = [int(x["index rank"]) for x in sorted_res]
    correct = [100 if x["ground truth"]==x["prediction"] else 0 for x in sorted_res]
    # sorter = zip(best_score, true_score, index_rank, correct)
    # sorter = sorted(sorter, key=operator.itemgetter(0), reverse=True)
    # best_score, true_score, index_rank, correct = zip(*sorter)
    plt.plot(true_score, "g", label="true score")
    plt.plot(best_score,"g--", label="best score")
    plt.plot(index_rank, label="index rank")
    # plt.plot(correct, label="correct")
    plt.axvspan(correct.index(0)-1.25,len(correct)-0.75, color="r", alpha=0.3, label="incorrect")
    plt.xticks(range(len(sorted_res)),[x["ground truth"] for x in sorted_res], rotation=-65)
    plt.legend()
    plt.title("spatial verification results")
    plt.savefig(output_folder+"/scores.png")
    plt.close()

def compare_historical_change(res, output_folder, sheetfile="data/blattschnitt_dr100_regular.geojson"):
    percent_content_query = [float(x["percent segmented"].replace(",",".")) for x in res]
    # get reference content
    import find_sheet, osm
    percent_content_reference = []
    for sheet in [x["ground truth"] for x in res]:
        bbox = find_sheet.find_poly_for_name(sheetfile,sheet)
        x,y = zip(*bbox)
        bbox = [min(x),min(y),max(x),max(y)]
        ref_path = os.path.join("E:/experiments/osm_drain/","rivers_%s.geojson" % "_".join(map(str,bbox)))
        with open(ref_path) as fr:
            data = json.load(fr)
            ref_img = osm.paint_features(data,bbox)
        num_blue_pixels_ref = cv2.countNonZero(ref_img)
        percent_blue_pixels_ref = num_blue_pixels_ref / (ref_img.shape[0] * ref_img.shape[1]) * 100
        percent_content_reference.append(percent_blue_pixels_ref)

    # sort
    score = [max(ast.literal_eval(x["scores"])) if x["ground truth"]==x["prediction"] else 0 for x in res]
    ranks = [int(x["index rank"]) for x in res]
    sorter = zip(score, percent_content_query, percent_content_reference, ranks)
    sorter = sorted(sorter, key=operator.itemgetter(0), reverse=True)
    score, percent_content_query, percent_content_reference, ranks = zip(*sorter)
    factor = [q/r for (q,r) in zip(percent_content_query,percent_content_reference)]
    # difference = [abs(q-r) for (q,r) in zip(percent_content_query,percent_content_reference)]
    # plot
    fig, axl = plt.subplots()
    axr = axl.twinx()
    axl.plot(percent_content_query, label="query")
    axl.plot(percent_content_reference, label="reference")
    axl.plot(factor, label="factor")
    # axl.plot(difference, label="difference")
    axl.legend(loc="upper left")
    axr.plot(score, "r", label="score")
    axr.plot(ranks, "k", label="ranks")
    axr.legend(loc="upper right")
    plt.savefig(output_folder+"/historical_change.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="path to results csv file")
    parser.add_argument("output", help="path for output evaluation files")
    parser.add_argument("--sheets", help="sheets json file path string", default=None)
    parser.add_argument("--editions", help="path to editions list", default=None)
    parser.add_argument("--coast", help="path to coast list", default=None)
    args = parser.parse_args()
    # py -3.7 eval/results_colourspace_lab.csv eval/eval_retrieval/ 
    # py -3.7 -m eval_scripts.eval_retrieval ../210510_geocv-kickoff/eval_result.csv ../210510_geocv-kickoff/indextest/ --sheets data/blattschnitt_dr100_regular.geojson --editions /e/data/deutsches_reich/SLUB/cut/editions.txt --coast /e/data/deutsches_reich/SLUB/cut/coast.txt

    results_file = args.results
    output_folder = args.output + "/figures/"
    os.makedirs(output_folder, exist_ok=True)

    res = get_retrieval_results(results_file)
    skipped = [x for x in res if "unknown" in (x["prediction"])]#.strip() == "unknown"]
    res = [x for x in res if (x["prediction"]).strip() != "unknown"]
    sheets_correct = [x["prediction"] for x in res if x["prediction"] == x["ground truth"]]
    sheets_incorrect = [x["ground truth"] for x in res if x["prediction"] != x["ground truth"]]

    # get index rank distribution before spatial verification
    print("%d sheets unreachable because of restrict number" % (len(skipped)))
    rank_compare(res, sheets_correct, sheets_incorrect, output_folder)

    # get % correct predictions
    print("%d of %d sheets correctly predicted" % (len(sheets_correct),len(res)+len(skipped)))

    # get RANSAC scores of correct predictions
    compare_scores(res, output_folder)

    # analyse Maha/Lowe
    maha_lowe(res, sheets_correct, sheets_incorrect, output_folder)

    # load special case lists
    if args.editions:
        # compare editions
        # editions_path = "E:/data/deutsches_reich/SLUB/cut/editions.txt"
        compare_special_cases(res, "editions",args.editions)

    if args.coast:
        # compare coast
        # coast_path = "E:/data/deutsches_reich/SLUB/cut/coast.txt"
        compare_special_cases(res, "coast",args.coast)

    # compare overedge
    # todo

    if args.sheets:
        # analyse amount of salient content and historical change (% blue pixels)
        compare_historical_change(res, output_folder, args.sheets)
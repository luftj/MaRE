import os
import argparse
import config
from eval_scripts.eval_helpers import get_georef_error, init
from main import process_list
from matplotlib import pyplot as plt
from operator import itemgetter

# list_file = "E:/data/deutsches_reich/SBB/cut/list_small.txt"
# annotations_file = "E:/data/deutsches_reich/SBB/cut/annotations.csv"
# sheets_file = "data/blattschnitt_dr100_regular.geojson"

def plot_error_bars(errors):
    # overview error bars
    median_error = list(errors.values())[len(errors)//2]
    mean_error = sum(errors.values())/len(errors)
    plt.bar(errors.keys(), errors.values(), 0.8, label="error")
    plt.axhline(mean_error, c="g", linestyle="--", label="mean")
    plt.annotate("%.0f" % mean_error,(0,mean_error + 30))
    plt.axhline(median_error, c="r", label="median")
    plt.annotate("%.0f" % median_error,(0,median_error + 30))
    plt.xticks(range(len(errors.keys())),errors.keys(),rotation=90)
    plt.legend()
    plt.xlabel("sheet")
    plt.ylabel("error [m]")
    plt.show()

def get_georef_error_snub(input_file, sheets_file, ground_truth_annotations_file, outpath):
    # get georef distances
    resultsfile = "%s/eval_georef_results.csv" % outpath
    from eval_scripts.eval_helpers import load_errors_csv
    errors = load_errors_csv(resultsfile)
    mean_error = sum(errors.values())/len(errors)
    return {"mean error": mean_error, "errors":errors}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("list", help="path to list of image files and truth sheet names")
    parser.add_argument("sheets", help="sheets json file path string")
    parser.add_argument("annotations", help="path to file containing ground truth corner annotations")
    parser.add_argument("output", help="path to output profiling files")
    parser.add_argument("--restrict", help="n most similar images to use in spatial verification", default=0)
    args = parser.parse_args()
    # example use:
    # py -3.7 -m eval_scripts.eval_register /e/data/deutsches_reich/wiki/highres/list_small.txt data/blattschnitt_dr100_regular.geojson /e/data/deutsches_reich/wiki/highres/annotations_wiki.csv eval/registration_eval/

    config.path_logs = args.output
    config.path_output = args.output
    init()

    # run exp
    # process_list(args.list, args.sheets, plot=False, img=True, restrict=0)

    # get error
    # res = get_georef_error(args.list, args.sheets, args.annotations, args.output)
    res = get_georef_error_snub(args.list, args.sheets, args.annotations, args.output)
    mean_error = res["mean error"] # get average precision
    errors = res["errors"]
    errors = dict(sorted(errors.items(),key=itemgetter(1)))
    print(errors)

    # get eval_logs for cross-checking w/ RANSAC score etc
    # todo
    # compare precision of correct/incorrect predictions
    # precision vs ransac score vs index rank -> estimate uncertainty
    # ECC score vs ECC iterations vs precision

    # make some fancy plots...
    plot_error_bars(errors)
    

    # special cases
    # load lists
    # coast
    # merged/overedge
    # editions

    # compare with state of the art

    # eval geodesy:
    # error vs latitude (shearing)
    # error vs distance to datum (projection deviation)
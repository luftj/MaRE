import os
import sys
import argparse
import config
import numpy as np
from eval_scripts.eval_helpers import get_georef_error, init, retrieval_results
from main import process_list
from matplotlib import pyplot as plt
from operator import itemgetter

def plot_error_bars(errors, outpath):
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
    plt.savefig(outpath + "/errors.png")
    plt.close()

def plot_error_geo(errors, sheetfile, outpath):
    import find_sheet

    bboxes = [find_sheet.find_bbox_for_name(sheetfile,name) for name in errors.keys()]
    #bbox = [minx, miny, maxx, maxy]
    coords = [[(b[0]-b[2])/2+b[0],(b[1]-b[3])/2+b[1]] for b in bboxes]

    import matplotlib.colors

    x = [c[0] for c in coords]
    y = [c[1] for c in coords]

    plt.scatter(x,y,c=list(errors.values()),cmap="Reds", label="error",norm=matplotlib.colors.LogNorm())
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.colorbar(label="log(error) [m]")
    plt.savefig(outpath + "/error_geo.png")
    plt.close()

def get_georef_error_snub(input_file, sheets_file, ground_truth_annotations_file, outpath):
    # get georef distances
    resultsfile = "%s/eval_georef_results.csv" % outpath
    from eval_scripts.eval_helpers import load_errors_csv
    errors = load_errors_csv(resultsfile)
    mean_error = sum(errors.values())/len(errors)
    return {"mean error": mean_error, "errors":errors}

def compare_special_cases(errors, specialcase_label, list_path, outpath, logfile=sys.stdout):
    cases = {}
    with open(list_path) as fr:
        for line in fr:
            line=line.strip()
            sheet, edition = line.split(",")
            cases[sheet] = edition
    possible_cases = list(set(cases.values()))

    # mean
    mean_error_total = sum(errors.values())/len(errors)
    print(f"total mean: {mean_error_total} ({len(errors)} sheets)", file=logfile)
    bars = [mean_error_total]
    for case in possible_cases:
        errors_case = [float(error) for sheet,error in errors.items() if cases[sheet] == case]
        mean_error_case = sum(errors_case)/len(errors_case)
        print(f"{case} mean: {mean_error_case} ({len(errors_case)} sheets)", file=logfile)
        bars.append(mean_error_case)
    labels = ["total"]+possible_cases
    plt.bar(labels,bars,label="mean")

    # median
    median_error_total = float(sorted(errors.values())[len(errors)//2])
    print(f"total median: {median_error_total} ({len(errors)} sheets)", file=logfile)
    bars = [median_error_total]
    for case in possible_cases:
        errors_case = [float(error) for sheet,error in errors.items() if cases[sheet] == case]
        median_error_case = sorted(errors_case)[len(errors_case)//2]
        print(f"{case} median: {median_error_case} ({len(errors_case)} sheets)", file=logfile)
        bars.append(median_error_case)
    plt.bar(labels,bars,label="median")

    plt.title(specialcase_label)
    plt.ylabel("mean error [m]")
    plt.legend()
    plt.savefig(outpath + "/errors_" + specialcase_label + ".png")
    plt.close()

def compare_state_of_the_art(errors, outpath):
    wm_maps=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t"]
    wm_km_sc_aff=[2.72,11.61,11.74,4.42,4.47,7.55,3.81,3.44,4.29,40.91,6.71,3.13,3.55,7.16,6.98,7.99,2.40,2.48,2.37,2.68]
    wm_px_sc_aff=[66.5,94.6,89.8,49.0,38.7,74.8,37.2,33.5,43.5,63.6,40.0,46.1,41.7,64.1,60.7,52.8,27.5,35.5,27.7,29.0]
    weinman_sc_tps=[61.3,93.3,81.8,33.8,32.5,74.5,34.0,29.4,42.8,56.7,41.9,50.5,53.1,51.4,43.1,59.8,27.7,26.5,26.2,27.8]

    wm_px_sc_aff_sorted = sorted(wm_px_sc_aff)
    weinman_affine_median_error = wm_px_sc_aff_sorted[len(wm_px_sc_aff_sorted)//2]
    weinman_affine_mean_error = sum(wm_px_sc_aff)/len(wm_px_sc_aff)
    print("Howe et al. affine median px",weinman_affine_median_error)
    print("Howe et al. affine mean px",weinman_affine_mean_error)
    
    weinman_sc_tps_sorted = sorted(weinman_sc_tps)
    weinman_tps_median_error = weinman_sc_tps_sorted[len(weinman_sc_tps_sorted)//2]
    weinman_tps_mean_error = sum(weinman_sc_tps)/len(weinman_sc_tps)
    print("Howe et al. TPS median px",weinman_tps_median_error)
    print("Howe et al. TPS mean px",weinman_tps_mean_error)

    ours_m_to_px = 1/6.4
    error_ours_px = [x*ours_m_to_px for x in errors.values()]
    print("ours median px", error_ours_px[len(error_ours_px)//2])
    print("ours mean px", sum(error_ours_px)/len(error_ours_px))

    plt.boxplot([wm_px_sc_aff_sorted,weinman_sc_tps,error_ours_px], vert=False, showmeans=True, medianprops={"color":"r"})
    plt.scatter([],[], c="g", marker="^", label="mean")
    plt.axhline(0, xmax=0, c="r", label="median")
    plt.xlabel("georeferencing RMSE [px]")
    plt.yticks([1,2,3,4],["Howe et al. (affine)","Howe et al. (TPS)","ours (affine)"])
    # plt.xticks(range(len(sheet_names_sorted)),sheet_names_sorted,rotation=-90)
    plt.legend()
    plt.title("Comparison with state of the art")
    plt.savefig(figurespath + "/compare_howe.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("list", help="path to list of image files and truth sheet names")
    parser.add_argument("sheets", help="sheets json file path string")
    parser.add_argument("annotations", help="path to file containing ground truth corner annotations")
    parser.add_argument("output", help="path to output profiling files")
    parser.add_argument("--editions", help="path to editions list", default=None)
    parser.add_argument("--coast", help="path to coast list", default=None)
    parser.add_argument("--restrict", help="n most similar images to use in spatial verification", default=0, type=int)
    args = parser.parse_args()
    # example use:
    # py -3.7 -m eval_scripts.eval_register /e/data/deutsches_reich/wiki/highres/list_regular.txt data/blattschnitt_dr100_regular.geojson /e/data/deutsches_reich/wiki/highres/annotations_wiki.csv /e/experiments/registration_eval/
    # --coast /e/data/deutsches_reich/wiki/highres/coast_list.txt --editions /e/data/deutsches_reich/wiki/highres/edition_list.txt

    config.path_logs = args.output
    config.path_output = args.output
    init()

    # run exp
    process_list(args.list, args.sheets, plot=False, img=True, restrict=args.restrict)

    # get error
    res = get_georef_error(args.list, args.sheets, args.annotations, args.output)
    # res = get_georef_error_snub(args.list, args.sheets, args.annotations, args.output)
    mean_error = res["mean error"] # get average precision
    errors = res["errors"]
    errors = dict(sorted(errors.items(),key=itemgetter(1)))

    # get retrieval results from eval_logs for cross-checking w/ RANSAC score etc
    retrieval_res = retrieval_results(args.output)["results"]
    retrieval_results_sorted = [ retrieval_res[s] for s in errors.keys()] # only look at maps with successful georeferencing
    
    # precision vs ransac score -> estimate uncertainty
    ransac = [x[1] for x in retrieval_results_sorted]
    plt.scatter(ransac,errors.values())
    # coef = np.polyfit(np.asarray(ransac), list(errors.values()), 2)
    # poly1d_fn = np.poly1d(coef)
    # plt.plot(ransac, poly1d_fn(ransac), "--k")
    plt.xlabel("RANSAC score")
    plt.ylabel("error [m]")
    plt.grid(True)
    plt.show()

    num_kps = [x[3] for x in retrieval_results_sorted]
    plt.scatter(num_kps,errors.values())
    # coef = np.polyfit(num_kps, list(errors.values()), 2)
    # poly1d_fn = np.poly1d(coef)
    # plt.plot(num_kps, poly1d_fn(num_kps), "--k")
    plt.xlabel("#keypoints")
    plt.ylabel("error [m]")
    plt.grid(True)
    plt.show()

    if args.restrict > 1:
        # precision vs index rank -> estimate uncertainty
        ranks = [x[0] for x in retrieval_results_sorted]
        plt.scatter(ranks, errors.values(), label="Index rank")
        # precision vs Mahalonobis -> estimate uncertainty
        maha = [x[2] for x in retrieval_results_sorted]
        plt.scatter(maha, errors.values(), label="Mahalonobis")

        plt.xlabel("Retrieval scores")
        plt.ylabel("error [m]")
        plt.legend()
        plt.grid(True)
        plt.show()

        # compare precision of correct/incorrect predictions
        # todo
    
    # ECC score vs ECC iterations vs precision

    # make some fancy plots...
    figurespath = args.output + "/figures/"
    os.makedirs(figurespath, exist_ok=True)
    plot_error_bars(errors, figurespath)

    # special cases
    # coast
    if args.coast:
        compare_special_cases(errors, "coast", args.coast, figurespath)

    # merged/overedge
    # todo

    # editions
    if args.editions:
        compare_special_cases(errors, "editions", args.editions, figurespath)

    # compare with state of the art
    compare_state_of_the_art(errors, figurespath)

    # eval geodesy:
    plot_error_geo(errors, args.sheets, figurespath)
    # error vs latitude (shearing)
    # error vs distance to datum (projection deviation)
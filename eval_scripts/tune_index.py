import os
from time import time

import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import importlib

import config

def run_exp(param_name, value, changeFunc, sheets_path_reference, list_path, out_path, rebuild_index = True, default_index=False):
    outpath = "%s/%s_%s" % (out_path, param_name, value)
    os.makedirs(outpath, exist_ok=True)

    # change settings/Parameters
    changeFunc(value)

    if rebuild_index or (not default_index):
        # change paths in config
        config.reference_sheets_path = outpath + "/sheets.clf"
        config.reference_index_path  = outpath + "/index.ann"

    import indexing
    importlib.reload(indexing)

    # rebuild index
    if rebuild_index:
        t0 = time()
        indexing.build_index(sheets_path_reference, store_desckp=False)
    
        # profile index building
        time_build = time()-t0
        print("time for building %f" % time_build)
        sheets = joblib.load(config.reference_sheets_path)
        n_ref_sheets = len(sheets)
        time_build /= n_ref_sheets
        print("per sheet: %f" % time_build)
    else:
        time_build = None
    
    t1 = time()
    # query index
    import annoytest
    importlib.reload(annoytest)
    lps = indexing.search_list(list_path)

    # profile lookup
    time_query = time()-t1
    print("time for querying %f" % time_query)
    with open(list_path) as f:
        n_query_sheets = sum(1 for _ in f)
    time_query /= n_query_sheets
    print("per sheet: %f" % time_query)

    with open(outpath+"/index_result.csv","w") as fp:
        for l,p in lps:
            fp.write("%s : %d\n"%(l,p))

    return time_build, time_query

def time_plots(times_build, times_query, param_to_tune, possible_values, figure_path):
    possible_values = list(map(str,possible_values))
    if times_build[0]:
        plt.bar(possible_values, times_build)
        plt.xlabel(param_to_tune)
        plt.ylabel("seconds per sheet")
        plt.title("time for building index")
        plt.savefig(figure_path+"/build_time.png")
        plt.close()
    plt.bar(possible_values, times_query)
    plt.xlabel(param_to_tune)
    plt.ylabel("seconds per sheet")
    plt.title("time for querying index")
    plt.savefig(figure_path+"/query_time.png")
    plt.close()

def plot_single_distribution(ranks,param_to_tune,val,figure_path):
    print("  0: %f%%"  % (len([x for x in ranks if x == 0])/len(ranks)*100))
    print("< 5: %f%%"  % (len([x for x in ranks if x < 5])/len(ranks)*100))
    print("< 10: %f%%"  % (len([x for x in ranks if x < 10])/len(ranks)*100))
    perc_20 = len([x for x in ranks if x < 20])/len(ranks)*100
    print("< 20: %f%%"  % (perc_20))
    perc_50 = len([x for x in ranks if x < 50])/len(ranks)*100
    print("< 50: %f%%"  % (perc_50))
    print("< 100: %f%%" % (len([x for x in ranks if x < 100])/len(ranks)*100))
    print("< mean: %f%%" % (len([x for x in ranks if x < sum(ranks)/len(ranks)])/len(ranks)*100))
    mean = sum(ranks)/len(ranks)
    print("mean rank: %f" % (mean))

    counts = np.bincount(ranks)
    counts = np.cumsum(counts)
    plt.plot(counts, label="rank distribution")
    plt.vlines(mean,counts[0],counts[-1], color="red", label="r<%d (mean)"%mean)
    plt.vlines(20,counts[0],counts[-1], color="orange", label="r<20 (%0.2f%%)"%perc_20)
    plt.vlines(50,counts[0],counts[-1], color="yellow", label="r<50 (%0.2f%%)"%perc_50)

    perc68 = np.searchsorted(counts,0.68*counts[-1])
    plt.vlines(perc68,counts[0],counts[-1], linestyles="--", label="r<%d (68%%)"%perc68)
    perc80 = np.searchsorted(counts,0.8*counts[-1])
    plt.vlines(perc80,counts[0],counts[-1], linestyles="-.", label="r<%d (80%%)"%perc80)
    perc90 = np.searchsorted(counts,0.9*counts[-1])
    plt.vlines(perc90,counts[0],counts[-1], linestyles="dotted", label="r<%d (90%%)"%perc90)
    perc99 = np.searchsorted(counts,0.99*counts[-1])
    plt.vlines(perc99,counts[0],counts[-1], linestyles=(0,(1,10)), label="r<%d (99%%)"%perc99)

    plt.xlabel("rank")
    plt.ylabel("# sheets")
    plt.title("rank distribution of %s %s" % (param_to_tune,val))
    plt.legend()
    plt.savefig(figure_path+"/desc_ranks_%s.png" % val)
    plt.close()

def result_overview(figure_path, results):
    for r in results:
        counts = np.bincount(r["ranks"])
        counts = np.cumsum(counts)
        plt.plot(counts, label=r["value"])
    plt.xlabel("rank")
    plt.ylabel("# sheets")
    plt.legend()
    plt.title("comparison of feature descriptors")
    plt.savefig(figure_path+"/desc_compare.png")
    plt.close()

def get_results(out_path, param_to_tune, possible_values):
    results = []
    for val in possible_values:
        outpath = "%s/%s_%s" % (out_path, param_to_tune, val)
        ranks = []
        with open(outpath+"/index_result.csv") as fr:
            for line in fr:
                sheet,rank = line.split(" : ")
                rank = int(rank)
                ranks.append(rank)
        results.append({"value":val,"ranks":ranks})
    return results

def run_and_eval(sheets, list, out_path, param_to_tune, possible_values, change_func):
    times_build = []
    times_query = []
    for val in possible_values:
        tb,tq = run_exp(param_to_tune, val, change_func, 
                        sheets, list, out_path,
                        rebuild_index=False, default_index=True)
        times_build.append(tb)
        times_query.append(tq)

    # load results from all runs
    results = get_results(out_path, param_to_tune, possible_values)

    figure_path = "%s/figures_%s" % (out_path, param_to_tune)
    os.makedirs(figure_path, exist_ok=True)

    print(results)
    # compare results
    result_overview(figure_path, results)

    # compare time taken for building and querying
    if len(times_build) > 0 and len(times_query) > 0:
        time_plots(times_build, times_query, param_to_tune, possible_values, figure_path)

    # details of best
    # todo: how to find "best"?
    for i in range(len(possible_values)):
        ranks = results[i]["ranks"]
        plot_single_distribution(ranks, param_to_tune, possible_values[i], figure_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("list", help="path to list of image files and truth sheet names")
    parser.add_argument("sheets", help="sheets json file path string")
    parser.add_argument("output", help="path to output evaluation files")
    args = parser.parse_args()
    # py -3.7 -m eval_scripts.tune_index E:/data/deutsches_reich/SBB/cut/list_med.txt data/blattschnitt_dr100_regular.geojson eval/tune_index/

    # param_to_tune = "index_annoydist"
    # possible_values = ["dot","euclidean"]
    # def changeAnnoyDist(val):
    #     config.index_annoydist = val
    # run_and_eval(args.sheets, args.list, args.out_path, param_to_tune, possible_values, changeAnnoyDist)
    
    # param_to_tune = "detector"
    # possible_values = ["surf_upright","kaze_upright","akaze_upright"]
    # def changeDetector(val):
    #     config.detector = val
    #     config.kp_detector = val
    # run_and_eval(args.sheets, args.list, args.out_path, param_to_tune, possible_values, changeDetector)

    param_to_tune = "n_descriptors_query"
    possible_values = [100,300,500,None]
    def changeDescQ(val):
        config.index_n_descriptors_query = val

    run_and_eval(args.sheets, args.list, args.out_path, param_to_tune, possible_values, changeDescQ)
import cProfile
import pstats
from pstats import SortKey

import os
import shutil
import logging
import ast
import re
import importlib
import argparse

import config
from eval_scripts.eval_helpers import init, results, load_errors_csv

def print_profile(prf_file="profile.prf", outfile=None, function_of_interest=""):
    if outfile:
        fw = open(outfile, "w")
        p = pstats.Stats(prf_file, stream=fw)
    else:
        p = pstats.Stats(prf_file)#.strip_dirs()
    # p.print_stats()
    # p.sort_stats(-1).print_stats()
    # p.sort_stats(SortKey.NAME)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(30)
    # p.sort_stats(SortKey.TIME).print_stats(30)
    # p.sort_stats(SortKey.TIME).print_stats(function_of_interest)
    # p.print_callers(1,"sort")
    if outfile:
        fw.close()

def read_time_calls(prf_file):
    with open(prf_file) as fr:
        lines= "".join(fr.readlines())
        total_time = float(re.search(r"[0-9.]+(?= seconds)",lines).group())
        func_values = re.findall(r"[0-9.]+(?=[\s\d.]+C:)",lines)
        # func_values = map(strip,func_values)
        # func_values = map(float,func_values)
        # func_values = list(func_values)
        func_ncalls = int(func_values[0])
        func_cumtime = float(func_values[3])
    return total_time, func_ncalls, func_cumtime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("list", help="path to list of image files and truth sheet names")
    parser.add_argument("sheets", help="sheets json file path string")
    parser.add_argument("annotations", help="path to file containing ground truth corner annotations")
    parser.add_argument("output", help="path to output profiling files")
    parser.add_argument("restrict", help="n most similar images to use in spatial verification. Set to 0 to profile registration.", type=int)
    args = parser.parse_args()

    # param_to_tune = "ransac_max_trials"
    # possible_values = [50,100,300,500,1000,1500,2000,3000]

    # param_to_tune = "ransac_random_state"
    # possible_values = [1,2,3,4]
    
    param_to_tune = "registration_ecc_iterations"
    possible_values = [50,100]

    # param_to_tune = "jpg_compression"
    # possible_values = [90,None]

    results_compare = []
    # shutil.rmtree("profiling", ignore_errors=True)
    outfolder = "%s/%s" % (args.output, param_to_tune)

    try:
        for val in possible_values:
            config.path_logs = "%s/logs_perf_test_%s/" % (outfolder, val)
            # config.path_output = "%s/profiling_%s_%s/" % (outfolder, param_to_tune, val)
            config.path_output = outfolder
            # os.makedirs(config.path_output, exist_ok=True)

            init()
            # change param
            # config.ransac_max_trials = val
            # config.ransac_random_state = val
            # if val:
            #     config.gdal_output_options += " -co NUM_THREADS=ALL_CPUS"
            # config.jpg_compression = val
            config.registration_ecc_iterations = val

            ### RUN
            from main import process_sheet, process_list
            # from segmentation import load_and_run
            cProfile.run('process_list(args.list,args.sheets,img=%s,restrict=args.restrict)'%(args.restrict==0),"%s/profile_%s.prf" % (outfolder, val))
            # cProfile.run('process_sheet(args.list,args.sheetse,img=%s,restrict=args.restrict)'%(args.restrict==0)',"%s/profile_%s.prf" % (outfolder, val))
            # cProfile.run('load_and_run(args.list,5)',"profile.prf")

            print_profile("%s/profile_%s.prf" % (outfolder,val),"%s/pr_out_%s.txt" % (outfolder, val), function_of_interest="georeference")
            total_time, func_ncalls, func_cumtime = read_time_calls("%s/pr_out_%s.txt" % (outfolder, val))

            if args.restrict==0:
                # profile registration
                # ground_truth_annotations_file = "E:/data/deutsches_reich/wiki/highres/annotations_wiki.csv"
                inputpath = os.path.dirname(args.list)#"E:/data/deutsches_reich/wiki/highres/"
                resultsfile = "%s/eval_georef_results_%s.csv" % (outfolder, val)

                import eval_georef
                importlib.reload(eval_georef) # make sure to reload config

                sheet_corners = eval_georef.read_corner_CSV(args.annotations)
                img_list = list(sheet_corners.keys())
                sheet_names, error_results, rmse_results = eval_georef.eval_list(img_list, sheet_corners, inputpath, args.sheets, config.path_output)
                eval_georef.dump_csv(sheet_names, error_results, rmse_results, outpath=resultsfile)

                errors = load_errors_csv(resultsfile)
                # errors = {"none":0}
                resultdict = {"value": val, "totaltime": total_time, "ncalls": func_ncalls, "cumtime": func_cumtime, "mean_error": sum(errors.values())/len(errors)}
            else:
                # profile retrieval
                score = results(outfolder)

                resultdict = {"value": val, "totaltime": total_time, "ncalls": func_ncalls, "cumtime": func_cumtime}
                resultdict.update(score)
                print(resultdict)
            results_compare.append(resultdict)
    finally:
        print(*results_compare, sep="\n")

    import matplotlib.pyplot as plt

    # plt.plot(possible_values,[x["ncalls"] for x in results_compare], label="# calls", color="black")
    plt.plot(possible_values,[x["totaltime"] for x in results_compare], label="total time [s]", color="green")
    # plt.tick_params(axis="y")
    plt.legend(loc="upper left")
    plt.xlabel(param_to_tune)
    plt.twinx()
    if args.restrict==0:
        plt.plot(possible_values,[x["mean_error"] for x in results_compare], label="avg. error")
    else:    
        plt.plot(possible_values,[x["score"] for x in results_compare], label="avg. score")
        plt.plot(possible_values,[x["#wrong"] for x in results_compare], label="# misses")
    # plt.tick_params(axis="y")
    # plt.xticks(range(0,possible_values[-1],50),possible_values)
    
    plt.legend(loc="upper right")
    plt.show()
import cProfile
import pstats
from pstats import SortKey

import os
import shutil
import logging
import ast
import re
import importlib

import config

def init():
    os.makedirs(config.path_logs, exist_ok=True)
    # os.makedirs(config.path_osm, exist_ok=True)
    os.makedirs(config.path_output, exist_ok=True)

    log = logging.getLogger()  # root logger - Good to get it only once.
    os.makedirs(config.path_logs, exist_ok=True)
    for hdlr in log.handlers[:]:  # remove the existing file handlers
        if isinstance(hdlr,logging.FileHandler):
            log.removeHandler(hdlr)
            
    logging.basicConfig(filename=(config.path_logs + '/prf.log') , 
                        level=logging.INFO, 
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-7.7s]  %(message)s") # gimme all your loggin'!

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

def results(logpath, results_file):
    from eval_logs import eval_logs
    # os.system("py -3.7 eval_logs.py > NUL")
    eval_logs(logpath)
    os.system("mv eval_result.csv %s" % results_file)
    prediction_results = {}
    with open(results_file) as fr:
        fr.readline() # skip header
        for line in fr:
            line = line.strip()
            vals = line.split("; ")
            gt = vals[0]
            pred = vals[1]
            scores = ast.literal_eval(vals[6])
            max_score = max(scores) if gt == pred else -1
            rank = int(vals[-1])
            prediction_results[gt] = (rank, max_score)

    scores = [x[1] for x in prediction_results.values()]
    num_incorrect = scores.count(-1)
    # print("prediction scores (index rank, ransac score)",prediction_results)
    avg_score = sum(scores)/len(prediction_results)
    return avg_score, num_incorrect

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

def load_errors_csv(filepath):
    with open(filepath) as fr:
        fr.readline()
        results = {}
        for line in fr:
            line.strip()
            sheet,mae,rmse = line.split("; ")
            results[sheet] = float(rmse)
    return results

if __name__ == "__main__":

    ### VARS
    # sheet = "319"
    # input_file = "E:/data/deutsches_reich/SLUB/cut/%s.png" % sheet
    # input_file = "E:/data/deutsches_reich/SLUB/cut/raw/%s.bmp" % sheet
    # input_file = "E:/data/deutsches_reich/SBB/cut/list.txt"
    # input_file = "E:/data/deutsches_reich/SBB/cut/list_small.txt"
    input_file = "E:/data/deutsches_reich/wiki/highres/list_small.txt"
    sheets_file = "data/blattschnitt_dr100_regular.geojson"
    restrict=0

    # param_to_tune = "ransac_max_trials"
    # possible_values = [50,100,300,500,1000,1500,2000,3000]

    # param_to_tune = "ransac_random_state"
    # possible_values = [1,2,3,4]
    
    param_to_tune = "registration_ecc_iterations"
    possible_values = [50,100,500,1000,3000]

    # param_to_tune = "jpg_compression"
    # possible_values = [90,None]

    results_compare = []
    # shutil.rmtree("profiling", ignore_errors=True)
    outfolder = "eval/profiling/%s" % (param_to_tune)

    try:
        for val in possible_values:
            config.path_logs = "%s/logs_perf_test_%s/" % (outfolder,val)
            config.path_output = "E:/experiments/profiling_%s_%s/" % (param_to_tune,val)
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
            cProfile.run('process_list(input_file,sheets_file,img=%s,restrict=restrict)'%(restrict==0),"%s/profile_%s.prf" % (outfolder, val))
            # cProfile.run('process_sheet(input_file,sheets_file,img=False,number=sheet,restrict=restrict)',"profiling/profile_%s.prf" % val)
            # cProfile.run('load_and_run(input_file,5)',"profile.prf")

            print_profile("%s/profile_%s.prf" % (outfolder,val),"%s/pr_out_%s.txt" % (outfolder, val), function_of_interest="georeference")
            total_time, func_ncalls, func_cumtime = read_time_calls("%s/pr_out_%s.txt" % (outfolder, val))

            if restrict==0:
                # profile registration
                ground_truth_annotations_file = "E:/data/deutsches_reich/wiki/highres/annotations_wiki.csv"
                inputpath = "E:/data/deutsches_reich/wiki/highres/"
                resultsfile = "%s/eval_georef_results_%s.csv" % (outfolder,val)

                import eval_georef
                importlib.reload(eval_georef) # make sure to reload config

                sheet_corners = eval_georef.read_corner_CSV(ground_truth_annotations_file)
                img_list = list(sheet_corners.keys())
                sheet_names, error_results, rmse_results = eval_georef.eval_list(img_list, sheet_corners, inputpath, sheets_file)
                eval_georef.dump_csv(sheet_names, error_results, rmse_results, outpath=resultsfile)

                errors = load_errors_csv(resultsfile)
                # errors = {"none":0}
                resultdict = {"value": val, "totaltime": total_time, "ncalls": func_ncalls, "cumtime": func_cumtime, "mean_error": sum(errors.values())/len(errors)}
            else:
                # profile retrieval
                avg_score, num_incorrect = results("profiling/logs_perf_test_%s/" % val, "profiling/results_%s.csv" % val)

                resultdict = {"value": val, "totaltime": total_time, "ncalls": func_ncalls, "cumtime": func_cumtime, "score": avg_score, "#wrong": num_incorrect}
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
    if restrict==0:
        plt.plot(possible_values,[x["mean_error"] for x in results_compare], label="avg. error")
    else:    
        plt.plot(possible_values,[x["score"] for x in results_compare], label="avg. score")
        plt.plot(possible_values,[x["#wrong"] for x in results_compare], label="# misses")
    # plt.tick_params(axis="y")
    # plt.xticks(range(0,possible_values[-1],50),possible_values)
    
    plt.legend(loc="upper right")
    plt.show()
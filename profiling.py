import cProfile
import pstats
from pstats import SortKey

import os
import shutil
import logging
import ast
import re

import config

def init():
    os.makedirs(config.path_logs, exist_ok=True)
    # os.makedirs(config.path_osm, exist_ok=True)
    # os.makedirs(config.path_output, exist_ok=True)

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
    # p.sort_stats(SortKey.CUMULATIVE).print_stats(30)
    p.sort_stats(SortKey.TIME).print_stats(function_of_interest)
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

    # print("prediction scores (index rank, ransac score)",prediction_results)
    avg_score = sum([x[1] for x in prediction_results.values()])/len(prediction_results)
    return avg_score

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

    ### VARS
    # sheet = "319"
    #input_file = "E:/data/deutsches_reich/SLUB/cut/%s.png" % sheet
    # input_file = "E:/data/deutsches_reich/SLUB/cut/raw/%s.bmp" % sheet
    input_file = "E:/data/deutsches_reich/SBB/cut/list.txt"
    input_file = "E:/data/deutsches_reich/SBB/cut/list_small.txt"
    sheets_file = "data/blattschnitt_dr100_regular.geojson"
    restrict=10

    param_to_tune = "ransac_max_trials"
    possible_values = [1000]

    # param_to_tune = "ransac_random_state"
    # possible_values = [1,2,3,4]

    results_compare = []
    # shutil.rmtree("profiling", ignore_errors=True)

    try:
        for val in possible_values:
            config.path_logs = "profiling/logs_perf_test_%s/" % val

            init()
            # change param
            config.ransac_max_trials = val
            # config.ransac_random_state = val

            ### RUN
            from main import process_sheet, process_list
            # from segmentation import load_and_run
            cProfile.run('process_list(input_file,sheets_file,5,plot=False,img=False,restrict=restrict)',"profiling/profile_%s.prf" % val)
            # cProfile.run('process_sheet(input_file,sheets_file,5,plot=False,img=False,number=sheet,restrict=restrict)',"profiling/profile_%s.prf" % val)
            # cProfile.run('load_and_run(input_file,5)',"profile.prf")

            print_profile("profiling/profile_%s.prf" % val,"profiling/pr_out_%s.txt" % val)
            total_time, func_ncalls, func_cumtime = read_time_calls("profiling/pr_out_%s.txt" % val)

            avg_score = results("profiling/logs_perf_test_%s/" % val, "profiling/results_%s.csv" % val)

            resultdict = {"value": val, "totaltime": total_time, "ncalls": func_ncalls, "cumtime": func_cumtime, "score": avg_score}
            print(resultdict)
            results_compare.append(resultdict)
    finally:
        print(*results_compare, sep="\n")

    import matplotlib.pyplot as plt

    # plt.plot(possible_values,[x["ncalls"] for x in results_compare], label="# calls", color="orange")
    plt.plot(possible_values,[x["totaltime"] for x in results_compare], label="total time", color="orange")
    # plt.tick_params(axis="y")
    plt.legend(loc="upper left")
    plt.twinx()
    plt.plot(possible_values,[x["score"] for x in results_compare], label="avg. score")
    # plt.tick_params(axis="y")
    # plt.xticks(range(0,possible_values[-1],50),possible_values)
    plt.xlabel(param_to_tune)
    plt.legend(loc="upper right")
    plt.show()
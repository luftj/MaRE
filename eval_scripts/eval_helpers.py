import ast
import os
import logging
import config
import time
import csv

from main import process_sheet, process_list
from eval_logs import eval_logs

def run_and_measure(input_file, sheets_file, out_path, param_to_tune, possible_values, change_param_func, restrict, scorefunc, scorefunc_params):
    results_compare = []

    for val in possible_values:
        outpath = "%s/%s_%s/" % (out_path, param_to_tune, val)
        config.path_output = outpath # not needed, since there are no output images
        config.path_logs = outpath

        init()
        change_param_func(val)
        
        # run
        t0 = time.time()
        process_list(input_file,sheets_file,plot=False,img=(restrict==0),restrict=restrict)
        # process_sheet(input_file,sheets_file,plot=False,img=False,number=sheet,restrict=restrict)
        total_time = time.time()-t0

        # measure
        score = scorefunc(*scorefunc_params, outpath=outpath)

        resultdict = {"value": val, "total_time": total_time}
        resultdict.update(score)
        
        print(resultdict)
        results_compare.append(resultdict)
    
    print(*results_compare, sep="\n")
    return results_compare

def init():
    os.makedirs(config.path_logs, exist_ok=True) # separate logs for eval_logs
    os.makedirs(config.path_output, exist_ok=True) # don't overwrite images from other experiments

    log = logging.getLogger()  # root logger - Good to get it only once.
    for hdlr in log.handlers[:]:  # remove the existing file handlers
        if isinstance(hdlr,logging.FileHandler):
            log.removeHandler(hdlr)
            
    logging.basicConfig(filename=(config.path_logs + '/prf.log') , 
                        level=logging.INFO, 
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-7.7s]  %(message)s") # gimme all your loggin'!

def results(outpath):
    # logpath = outpath
    results_file = outpath + "/retrieval_results.csv"
    eval_logs(outpath, results_file)
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
    print("prediction scores (index rank, ransac score)",prediction_results)
    avg_score = sum(scores)/len(prediction_results)
    return {"score":avg_score, "wrong": num_incorrect}

def retrieval_results(outpath):
    # logpath = outpath
    results_file = outpath + "/retrieval_results.csv"
    eval_logs(outpath, results_file)
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
            mahalonobis = float(vals[12])
            num_kps = int(vals[7])
            prediction_results[gt] = (rank, max_score, mahalonobis, num_kps)

    scores = [x[1] for x in prediction_results.values()]
    num_incorrect = scores.count(-1)
    print("prediction scores (index rank, ransac score)",prediction_results)
    avg_score = sum(scores)/len(prediction_results)
    return {"score":avg_score, "wrong": num_incorrect, "results": prediction_results}

def save_results(results, path):
    if len(results) == 0:
        print("no results")
        exit()

    with open('path', 'w') as f:  
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for elem in results:
            writer.writerow(elem)

def load_errors_csv(filepath):
    with open(filepath) as fr:
        fr.readline()
        results = {}
        for line in fr:
            line.strip()
            sheet,mae,rmse = line.split("; ")
            results[sheet] = float(rmse)
    return results

def get_georef_error(input_file, sheets_file, ground_truth_annotations_file, outpath):
    # get georef distances
    inputpath = os.path.dirname(input_file)
    resultsfile = "%s/eval_georef_results.csv" % outpath
    import eval_georef
    sheet_corners = eval_georef.read_corner_CSV(ground_truth_annotations_file)
    img_list = list(sheet_corners.keys())
    sheet_names, error_results, rmse_results = eval_georef.eval_list(img_list, sheet_corners, inputpath, sheets_file, config.path_output)
    eval_georef.dump_csv(sheet_names, error_results, rmse_results, outpath=resultsfile)
    errors = load_errors_csv(resultsfile)
    mean_error = sum(errors.values())/len(errors)
    return {"mean error": mean_error, "errors":errors}
import ast
import os
import logging
import config

def init():
    os.makedirs(config.path_logs, exist_ok=True) # separate logs for eval_logs
    # os.makedirs(config.path_osm, exist_ok=True)
    os.makedirs(config.path_output, exist_ok=True) # don't overwrite images from other experiments

    log = logging.getLogger()  # root logger - Good to get it only once.
    for hdlr in log.handlers[:]:  # remove the existing file handlers
        if isinstance(hdlr,logging.FileHandler):
            log.removeHandler(hdlr)
            
    logging.basicConfig(filename=(config.path_logs + '/prf.log') , 
                        level=logging.INFO, 
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-7.7s]  %(message)s") # gimme all your loggin'!

def results(logpath, results_file):
    from eval_logs import eval_logs
    eval_logs(logpath, results_file)
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

def save_results(results, path):
    import csv
    
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
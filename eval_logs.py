from os import listdir
from os.path import isfile, join
import re
import json
from math import sqrt

from config import path_logs

def dump_csv(experiments, resultpath):
    print("writing to file...")
    with open("%s.csv" % resultpath, "w", encoding="utf-8") as eval_fp:
        # header
        eval_fp.write("ground truth; prediction; ground truth position; georef success; avg time per sheet; times; scores; number of detected keypoints; template scores; registration time; command; percent segmented; mahalanobis; Lowe's test ratio; index rank\n")

        for exp in experiments.values():
            try:
                eval_fp.write("%s; %s; %d; %s; %.2f; %s; %s; %d; %s; %.2f; %s; %s; %.2f; %.2f; %d\n" % (exp["ground_truth"],
                                                                exp["prediction"],
                                                                exp["gt_pos"],
                                                                exp["georef_success"],
                                                                exp.get("avg_time",-1),
                                                                exp.get("times",[]),
                                                                exp.get("scores",[]),
                                                                exp.get("num_keypoints",-1),
                                                                exp.get("template_scores","[]"),
                                                                exp.get("register_time",-1), # might not have been registered
                                                                exp["command"],
                                                                exp.get("percent_segmented",-1).replace(".",","),
                                                                exp.get("mahalanobis",-1),
                                                                exp.get("lowes_ratio",-1),
                                                                exp.get("index_rank",-2)))
            except KeyError as e:
                print(e)
                print("skipping exp for %s" % exp.get("ground_truth",None))

def dump_json(experiments,resultpath):
    with open("%s.json" % resultpath, "w", encoding="utf-8") as eval_fp:
        json.dump(experiments, eval_fp)

def mahalanobis_distance(scores):
    max_val = max(scores)
    sample = scores.copy()
    sample.remove(max_val)
    if len(sample) == 0:
        return -1
    mean = sum(sample)/len(sample)
    if mean == max_val:
        return 0
    if len(sample) == 1:
        return 0
    sd = sqrt(sum([(x-mean)**2 for x in sample])/(len(sample)-1)) # sample standard deviation

    # print("mean",mean)
    # print("standard deviation",sd)

    if sd == 0:
        dist = 0
    else:
        dist = (max_val - mean) / sd
    # print("distance", dist)
    return dist

def lowes_ratio(scores):
    if len(scores) <= 1:
        return -1
    values = scores.copy()
    maxval = max(values)
    if maxval == 0:
        return 0
    values.remove(maxval)
    max_noise = max(values)
    return max_noise/maxval


def plot_score_dist(x):
    import matplotlib.pyplot as plt
    import numpy as np

    def gaussian(x, mu, sig, scale):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * scale

    dist=mahalanobis_distance(x)

    sample = x.copy()
    sample.remove(max(x))
    mean = sum(sample)/len(sample)
    sd = sqrt(sum([(o-mean)**2 for o in sample])/(len(sample)-1))

    x_values = np.linspace(min(x),max(x))
    plt.plot(x_values, gaussian(x_values, mean, sd, x.count(max(x,key=x.count))), label="dist")
    plt.axvline(mean + sd, c="r",linestyle="--", label="1σ")
    plt.axvline(mean + 2*sd, c="r",linestyle=":", label="2σ")
    plt.axvline(mean + 3*sd, c="r",linestyle="-.", label="3σ")
    plt.axvline(mean + 4*sd, c="r",linestyle="-", label="4σ")
    print("4s",mean+4*sd)
    # plt.axvline(max(x), c="y",linestyle="-", label="dist")
    plt.hist(x,max(x)+2, label="obs", align="mid")
    plt.xticks(range(min(x)-1,max(x)+1))
    plt.legend()
    plt.show()
    exit()

def eval_logs(logpath, resultpath="eval_result"):
    # plot_score_dist([6, 9, 14, 9, 12, 8, 7, 11, 7, 9, 10, 8, 10, 9, 7, 10, 9, 9, 7, 8, 9])
    # plot_score_dist( [4, 3, 3, 3, 3, 4, 4, 3, 4, 4, 5, 4, 5, 4, 5, 4, 3, 4, 5, 4, 4])
    # plot_score_dist( [8, 9, 7, 6, 5, 11, 5, 8, 11, 9, 15, 5, 7, 7, 6, 6, 4, 6, 8, 5, 7])
    # plot_score_dist( [4, 5, 6, 6, 4, 4, 5, 4, 6, 5, 9, 6, 5, 4, 5, 6, 4, 5, 5, 5, 5])
    # plot_score_dist( [4, 5, 6, 6, 4, 4, 5, 4, 6, 5, 9, 6, 5, 4, 5, 6, 4, 5, 5, 5, 5])

    log_files = [f for f in listdir(logpath) if isfile(join(logpath, f)) and f.endswith(".log")]


    experiments = {}

    for log_file in log_files:
        print(log_file)
        file_path = join(logpath, log_file)

        experiment_data = None
        sum_template_score = 0
        command = ""

        with open(file_path) as log_fp:

            for line in log_fp:
                if not "[INFO" in line:
                    continue

                # strip timestamp, thread, \n, etc...
                line = line[48:-1]

                # print(line)

                if "Processing file" in line:
                    # end previous entry
                    if experiment_data:
                        if "times" in experiment_data:
                            experiment_data["avg_time"] = sum(experiment_data["times"])/len(experiment_data["times"])

                        if "scores" in experiment_data:
                            experiment_data["mahalanobis"] = mahalanobis_distance(experiment_data["scores"])
                            experiment_data["lowes_ratio"] = lowes_ratio(experiment_data["scores"])

                        if "ground_truth" in experiment_data:

                            experiments[experiment_data["ground_truth"]] = experiment_data
                            print("end for gt", experiment_data["ground_truth"])

                    # start new entry
                    experiment_data = {}
                    experiment_data["georef_success"] = False
                    experiment_data["command"] = command

                # get experiment command for reproducing
                if "new experiment with" in line:
                    s = line.split("with: ")[-1].replace("'","\"")
                    l = json.loads(s)
                    command = " ".join(l)

                # get ground truth
                # get result
                elif "result:" in line:
                    pred = re.search(r"(?<=pred:)\s*[^,'\s]*", line)[0]
                    experiment_data["prediction"] = pred.strip()
                    gt = re.search(r"(?<=gt:)\s*[^,'\s]*", line)[0]
                    experiment_data["ground_truth"] = gt.strip()

                elif "template matching score" in line:
                    sum_template_score += float(line.split(":")[-1])

                elif "Truth at position" in line:
                    experiment_data["index_rank"] = int(line.replace("Truth at position ","").replace(" in index.",""))

                # get distance distribution
                elif "target" in line:
                    if not "scores" in experiment_data:
                        experiment_data["scores"] = []
                        experiment_data["times"] = []
                        experiment_data["template_scores"] = []
                    score = int(re.search(r"(?<=Score\s)[0-9]*", line)[0])
                    experiment_data["scores"].append(score)

                    time = float(re.search(r"(?<=time:\s)[0-9]*\.[0-9]*", line)[0])
                    experiment_data["times"].append(time)

                    if "num_keypoints" in experiment_data:
                        if experiment_data["num_keypoints"] == 0:
                            avg_template_score = -1
                        else:
                            avg_template_score = sum_template_score/experiment_data["num_keypoints"]
                        experiment_data["template_scores"].append(avg_template_score)
                    sum_template_score = 0

                # get distance distribution
                elif "ground truth at position" in line:
                    experiment_data["gt_pos"] = int(line.split(" ")[-1])

                # get georef success
                elif "saved georeferenced file" in line:
                    experiment_data["georef_success"] = True

                # get georef time
                elif "s for registration" in line:
                    time = float(re.search(r"(?<=time:\s)[0-9]*\.[0-9]*", line)[0])
                    experiment_data["register_time"] = time
                    
                # get num interest points
                elif "number of corners" in line or "number of used keypoints" in line:
                    experiment_data["num_keypoints"] = int(line.split(" ")[-1])

                # get segmented pixels
                elif "segmented" in line:
                    experiment_data["percent_segmented"] = (line.split(" ")[-2])

        
        # end entry
        if experiment_data:
            if "times" in experiment_data:
                experiment_data["avg_time"] = sum(experiment_data["times"])/len(experiment_data["times"])

            if "scores" in experiment_data:
                experiment_data["mahalanobis"] = mahalanobis_distance(experiment_data["scores"])
                experiment_data["lowes_ratio"] = lowes_ratio(experiment_data["scores"])

            if "ground_truth" in experiment_data:
                experiments[experiment_data["ground_truth"]] = experiment_data
                print("end for gt", experiment_data["ground_truth"])

    resultpath = resultpath.split(".")[0]
    dump_csv(experiments, resultpath)
    dump_json(experiments, resultpath)

if __name__ == "__main__":
    eval_logs(path_logs)
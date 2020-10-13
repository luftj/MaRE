from os import listdir
from os.path import isfile, join
import re
import json

from config import path_logs

def dump_csv(experiments):
    print("writing to file...")
    with open("eval_result.csv", "w", encoding="utf-8") as eval_fp:
        # header
        eval_fp.write("ground truth; prediction; ground truth position; georef success; avg time per sheet; times; scores; number of detected keypoints; template scores; registration time; command; percent segmented\n")

        for exp in experiments.values():
            try:
                eval_fp.write("%s; %s; %d; %s; %.2f; %s; %s; %d; %s; %.2f; %s; %s\n" % (exp["ground_truth"],
                                                                exp["prediction"],
                                                                exp["gt_pos"],
                                                                exp["georef_success"],
                                                                exp["avg_time"],
                                                                exp["times"],
                                                                exp["scores"],
                                                                exp["num_keypoints"],
                                                                exp.get("template_scores","[]"),
                                                                exp.get("register_time",-1), # might not have been registered
                                                                exp["command"],
                                                                exp.get("percent_segmented",-1).replace(".",",")))
            except KeyError as e:
                print(e)
                print("skipping exp for %s" % exp.get("ground_truth",None))

def dump_json(experiments):
    with open("eval_result.json", "w", encoding="utf-8") as eval_fp:
        json.dump(experiments, eval_fp)

if __name__ == "__main__":
    log_files = [f for f in listdir(path_logs) if isfile(join(path_logs, f))]


    experiments = {}

    for log_file in log_files:
        print(log_file)
        file_path = join(path_logs, log_file)

        experiment_data = None
        sum_template_score = 0

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
                    pred = re.search(r"(?<=pred:)[^,']*", line)[0]
                    experiment_data["prediction"] = pred
                    gt = re.search(r"(?<=gt:)[^,']*", line)[0]
                    experiment_data["ground_truth"] = gt

                elif "template matching score" in line:
                    sum_template_score += float(line.split(":")[-1])

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

            if "ground_truth" in experiment_data:
                experiments[experiment_data["ground_truth"]] = experiment_data
                print("end for gt", experiment_data["ground_truth"])


    dump_csv(experiments)
    dump_json(experiments)
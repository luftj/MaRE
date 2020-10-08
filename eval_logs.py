from os import listdir
from os.path import isfile, join
import re

if __name__ == "__main__":
    logs_path = "logs/"
    log_files = [f for f in listdir(logs_path) if isfile(join(logs_path, f))]


    experiments = {}

    for log_file in log_files:
        file_path = join(logs_path, log_file)

        experiment_data = {}
        experiment_data["georef_success"] = False

        with open(file_path, encoding="utf-8") as log_fp:

            for line in log_fp:
                if not "[INFO" in line:
                    continue

                # strip timestamp, thread, \n, etc...
                line = line[48:-1]

                # print(line)

                # get experiment command for reproducing
                if "new experiment with" in line:
                    experiment_data["command"] = line.split("with: ")[-1]

                # get ground truth
                elif "with gt:" in line:
                    experiment_data["ground_truth"] = line.split("with gt: ")[-1]

                # get result
                elif "result:" in line:
                    experiment_data["prediction"] = line.split(",")[1][2:-1].replace("pred:","")

                # get distance distribution
                elif "target" in line:
                    if not "scores" in experiment_data:
                        experiment_data["scores"] = []
                        experiment_data["times"] = []
                    score = int(re.search(r"(?<=Score\s)[0-9]*", line)[0])
                    experiment_data["scores"].append(score)

                    time = float(re.search(r"(?<=time:\s)[0-9]*\.[0-9]*", line)[0])
                    experiment_data["times"].append(time)

                # get distance distribution
                elif "ground truth at position" in line:
                    experiment_data["gt_pos"] = int(line.split(" ")[-1])

                    
                # get georef success
                elif "saved georeferenced file" in line:
                    experiment_data["georef_success"] = True

        if "times" in experiment_data:
            experiment_data["avg_time"] = sum(experiment_data["times"])/len(experiment_data["times"])

        experiments[experiment_data["ground_truth"]] = experiment_data

    with open("eval_result.csv", "w", encoding="utf-8") as eval_fp:
        print("writing to file...")
        # header
        eval_fp.write("ground truth; prediction; gt pos; georef success; avg_time; times; scores; command\n")

        for exp in experiments.values():
            print(exp)
            try:
                eval_fp.write("%s; %s; %d, %s, %f, %s, %s, %s\n" % (exp["ground_truth"],
                                                                exp["prediction"],
                                                                exp["gt_pos"],
                                                                exp["georef_success"],
                                                                exp["avg_time"],
                                                                exp["times"],
                                                                exp["scores"],
                                                                exp["command"]))
            except KeyError as e:
                print(e)
                print("skipping exp for %s", exp.get("ground_truth",None))
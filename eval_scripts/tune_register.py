import os
# import logging
import argparse
# import time

import config
from eval_scripts.eval_helpers import init, save_results, load_errors_csv, run_and_measure, get_georef_error

def run_experiment(input_file, sheets_file, ground_truth_annotations_file, out_path, param_to_tune, possible_values, change_param_func):
    results_compare = run_and_measure(input_file, sheets_file, out_path, 
                            param_to_tune, possible_values, change_param_func, 
                            0, get_georef_error, [input_file, sheets_file, ground_truth_annotations_file])
    
    save_results(results_compare,"%s/%s.csv" % (out_path, param_to_tune))
    results_compare_sorted = sorted(results_compare, key=lambda x: x["mean error"])
    print("best value: %s, with mean RMSE %f" % (results_compare_sorted[0]["value"], results_compare_sorted[0]["mean error"]))#, results_compare_sorted[1]["mean error"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("list", help="path to list of image files and truth sheet names")
    parser.add_argument("sheets", help="sheets json file path string")
    parser.add_argument("annotations", help="path to file containing ground truth corner annotations")
    parser.add_argument("output", help="path to output profiling files")
    args = parser.parse_args()
    # example use:
    # py -3.7 -m eval_scripts.tune_register /e/data/deutsches_reich/wiki/highres/list_small.txt data/blattschnitt_dr100_regular.geojson /e/data/deutsches_reich/wiki/highres/annotations_wiki.csv eval/registration_tuning/

    # param_to_tune = "closing"
    # possible_values = [(11,11),(19,19)]

    # def change_closing_func(val):
    #     config.segmentation_closingkernel = val

    # param_to_tune = "registration_mode"
    # possible_values = ["ransac","ecc","both"]

    # def change_regmode_func(val):
    #     config.registration_mode = val

    # param_to_tune = "registration_ecc_iterations"
    # possible_values = [50,100,500,1000,3000]

    # def change_eccit_func(val):
    #     config.registration_ecc_iterations = val

    param_to_tune = "registration_ecc_eps"
    possible_values = [1e-4,1e-5]

    def change_ecceps_func(val):
        config.registration_ecc_eps = val

    run_experiment(args.list, args.sheets, args.annotations, args.output,
                    param_to_tune, possible_values, change_ecceps_func)
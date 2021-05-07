import os
import logging
import argparse
import time

import config
from eval_scripts.eval_helpers import init, save_results, load_errors_csv

def run_experiment(input_file, sheets_file, ground_truth_annotations_file, outpath, param_to_tune, possible_values, change_param_func):
    results_compare = []

    for val in possible_values:
        config.path_logs = "%s/logs_%s_%s/" % (outpath, param_to_tune, val)

        outfolder = "%s/%s_%s" % (outpath, param_to_tune, val)
        config.path_output = outfolder + "/results/"
        init()

        change_param_func(val)
        ### RUN
        from main import process_sheet, process_list
        t0 = time.time()
        process_list(input_file,sheets_file,restrict=0)
        total_time = time.time()-t0
        
        # get georef distances
        inputpath = os.path.dirname(input_file)
        resultsfile = "%s/eval_georef_results.csv" % outfolder
        import eval_georef
        sheet_corners = eval_georef.read_corner_CSV(ground_truth_annotations_file)
        img_list = list(sheet_corners.keys())
        sheet_names, error_results, rmse_results = eval_georef.eval_list(img_list, sheet_corners, inputpath, sheets_file, config.path_output)
        eval_georef.dump_csv(sheet_names, error_results, rmse_results, outpath=resultsfile)
        errors = load_errors_csv(resultsfile)

        resultdict = {"value": val, "mean_error": sum(errors.values())/len(errors), "total_time": total_time, "errors": errors}
        print(resultdict)
        results_compare.append(resultdict)
    
    print(*results_compare, sep="\n")
    # save_results(results_compare,"eval/tune_register/%s.csv" % param_to_tune)
    results_compare_sorted = sorted(results_compare, key=lambda x: x["mean_error"])
    # # print(results_compare_sorted)
    print("best value: %s, with mean RMSE %f" % (results_compare_sorted[0]["value"], results_compare_sorted[0]["mean_error"]))#, results_compare_sorted[1]["mean_error"]))

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
    possible_values = [1e-2,1e-3,1e-4,1e-5]

    def change_ecceps_func(val):
        config.registration_ecc_eps = val

    run_experiment(args.list, args.sheets, args.annotations, args.output,
                    param_to_tune, possible_values, change_ecceps_func)
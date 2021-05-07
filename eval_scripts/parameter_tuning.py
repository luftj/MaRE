import os
# import logging
# import ast
import argparse

import config
from eval_scripts.eval_helpers import init, results, save_results, run_and_measure

def run_experiment(input_file, sheets_file, out_path, restrict, param_to_tune, possible_values, change_param_func):
    results_compare = run_and_measure(input_file, sheets_file, out_path, 
                            param_to_tune, possible_values, change_param_func, 
                            restrict, results, [])

    save_results(results_compare,"%s/%s.csv" % (out_path, param_to_tune))
    results_compare_sorted = sorted(results_compare, key=lambda x: x["score"], reverse=True)
    
    print("best value: %s, with score %f (next best: %f)" % (results_compare_sorted[0]["value"], results_compare_sorted[0]["score"], results_compare_sorted[1]["score"]))

def test_osm(input_file, sheets_file, out_path, restrict):
    # TODO: untested
    osm_values = {
        "full" : """[out:json];
                (nwr ({{bbox}}) [water=lake]; 
                way ({{bbox}}) [natural=water] [name]; 
                way ({{bbox}}) [type=waterway] [name]; 
                way ({{bbox}}) [waterway=river] [name];
                way ({{bbox}}) [waterway=canal] [name];
                way ({{bbox}}) [water=river];
                way ({{bbox}}) [waterway=stream] [name];
                way ({{bbox}}) [natural=coastline];
                way ({{bbox}}) [waterway=ditch];
                way ({{bbox}}) [waterway=drain];
                );
                out body;
                >;
                out skel qt;""",
        "river_only" : """[out:json];
                (way ({{bbox}}) [natural=water] [name]; 
                way ({{bbox}}) [type=waterway] [name]; 
                way ({{bbox}}) [waterway=river] [name];
                way ({{bbox}}) [water=river];
                );
                out body;
                >;
                out skel qt;"""
    }

    def change_func(val):
        # change param
        config.osm_query = osm_values[val]
        config.force_osm_download = True
        config.path_osm = "%s/osm_exp_%s/" % (out_path, val)
        
        os.makedirs(config.path_osm, exist_ok=True)
        # rebuild index
        config.reference_index_path = "%s/index/index_%s.ann" % (out_path, val)
        config.reference_descriptors_folder = "%s/index/descriptors/desc_%s.bin" % (out_path, val)
        config.reference_keypoints_folder = "%s/index/keypoints/desc_%s.bin" % (out_path, val)
        os.makedirs(config.reference_descriptors_folder, exist_ok=True)
        os.makedirs(config.reference_keypoints_folder, exist_ok=True)
        from indexing import build_index
        build_index(sheets_file)

    run_experiment(input_file, sheets_file, out_path, restrict, "osm_query", osm_values, change_func)

def test_segmentation(input_file, sheets_file, out_path, restrict):
    # blurring
    blur_values = [(17,17),(19,19),(21,21)]
    def change_func(val):
        config.segmentation_blurkernel = val
    run_experiment(input_file, sheets_file, out_path, restrict, "blur_kernel", blur_values, change_func)

    # opening
    # opening_values = [(3,3),(5,5),(7,7),(9,9)]
    # def change_func2(val):
    #     config.segmentation_openingkernel = val
    # run_experiment("opening_kernel", opening_values, change_func2)

def test_hsvlab(input_file, sheets_file, out_path, restrict):
    values = ["hsv","lab"]

    def change_func(val):
        config.segmentation_colourspace = val
        config.codebook_response_threshold = False
        if val=="hsv":
            # config.segmentation_blurkernel = (9,9)
            config.segmentation_colourbalance_percent = 3
            config.segmentation_lowerbound = (120,  0,  90)
            config.segmentation_upperbound = (255, 255, 255)
        elif val=="lab":
            # config.segmentation_blurkernel = (19,19)
            config.segmentation_colourbalance_percent = 5
            config.segmentation_lowerbound = (0,0,10)
            config.segmentation_upperbound = (255, 90, 100)
    
    run_experiment(input_file, sheets_file, out_path, restrict, "colourspace", values, change_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("list", help="path to list of image files and truth sheet names")
    parser.add_argument("sheets", help="sheets json file path string")
    parser.add_argument("output", help="path to output profiling files")
    parser.add_argument("--restrict", help="n most similar images to use in spatial verification", default=30)
    args = parser.parse_args()
    # run with $ py -3.7 -m eval_scripts.parameter_tuning /e/data/deutsches_reich/wiki/highres/list_small.txt data/blattschnitt_dr100_regular.geojson /e/experiments/paramtunetesthsv/

    # needs local OSM data, otherwise it takes too long

    # test different OSM queries
    # test_osm(args.list, args.sheets, args.output, args.restrict)
    
    # test different segmentation parameters
    # test_segmentation(args.list, args.sheets, args.output, args.restrict)

    # test hsv vs lab segmentation
    test_hsvlab(args.list, args.sheets, args.output, args.restrict)

    print("finished all experiments!")
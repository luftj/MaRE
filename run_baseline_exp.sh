#!/bin/bash

input="/e/data/deutsches_reich/SLUB/cut/list.txt"
sheets="data/blattschnitt_dr100_regular.geojson"

# py -3.7 indexing.py \
#    --rebuild \
#     $sheets

# for osm baseline, we need to skip the segmentation
# py -3.7 make_osm_baseline.py $sheets $input /e/data/deutsches_reich/osm_baseline/
#py -3.7 make_osm_baseline.py data/blattschnitt_dr100_regular.geojson /e/data/deutsches_reich/SLUB/cut/list.txt /e/data/deutsches_reich/osm_baseline/
# exit
input="/e/data/deutsches_reich/osm_baseline/list.txt"

# py -3.7 indexing.py \
#     --list $input \
#     $sheets 
# mv index_result.csv icc_eval/index_result_osm_baseline.csv
# exit
# py -3.7 main.py \
#     $input \
#     $sheets \
#     -r 1 \
#     -v \
#     --noimg
# # py -3.7 test_osm_baseline.py $input $sheets -r 20

# py -3.7 eval_logs.py
# mv eval_result.csv icc_eval/eval_results_osm_baseline.csv
# exit
# degraded osm
# py -3.7 make_osm_baseline.py $sheets $input /e/data/deutsches_reich/osm_baseline_circles500_15-50/ --circles 500

input="/e/data/deutsches_reich/osm_baseline_circles500_15-50/list.txt"

# py -3.7 indexing.py \
#     --list $input \
#     $sheets 
# mv index_result.csv icc_eval/index_result_osm_baseline_circles500_15-50.csv
# exit
py -3.7 main.py \
    $input \
    $sheets \
    -r 30 \
    --noimg

py -3.7 eval_logs.py
mv eval_result.csv icc_eval/eval_results_osm_baseline_circles500_15-50.csv
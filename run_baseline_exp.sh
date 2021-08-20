#!/bin/bash

# input="/e/data/deutsches_reich/SLUB/cut/list.txt"
# sheets="data/blattschnitt_dr100_regular.geojson"
input="/e/data/dr500/list.txt"
sheets="/e/data/dr500/blattschnitt_kdr500_wgs84.geojson"
series="dr500coarse"

input="/e/data/usgs/100k/list.txt"
sheets="/e/data/usgs/indices/CellGrid_30X60Minute.json"
series="usgs100"
# py -3.7 indexing.py \
#    --rebuild \
#     $sheets

# for osm baseline, we need to skip the segmentation
#py -3.7 make_osm_baseline.py $sheets $input /e/data/deutsches_reich/osm_baseline/
# py -3.7 make_osm_baseline.py $sheets $input /e/data/$series/osm_baseline/
# exit
#py -3.7 make_osm_baseline.py data/blattschnitt_dr100_regular.geojson /e/data/deutsches_reich/SLUB/cut/list.txt /e/data/deutsches_reich/osm_baseline/
# exit
# input="/e/data/deutsches_reich/osm_baseline/list.txt"
input="/e/data/$series/osm_baseline/list.txt"

py -3.7 indexing.py \
    --list $input \
    $sheets 
mv index_result.csv icc_eval/index_result_$series"_osm_baseline.csv"
exit
py -3.7 main.py \
    $input \
    $sheets \
    -r 10 \
    --noimg

py -3.7 eval_logs.py
mv eval_result.csv icc_eval/eval_results_$series"_osm_baseline.csv"
exit
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
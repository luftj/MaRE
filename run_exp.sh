#!/bin/bash

# input="/e/data/deutsches_reich/SLUB/cut/raw/list.txt"
#sheets="data/blattschnitt_dr100_merged.geojson"
# sheets="data/blattschnitt_dr100_regular.geojson"

input="/e/data/dr500/list_manual.txt"
sheets="/e/data/dr500/blattschnitt_kdr500_wgs84.geojson"
series="dr500idxw1000_manual"
# input="/e/data/usgs/100k/unset_segmented/list_filtered.txt"
# sheets="/e/data/usgs/indices/CellGrid_30X60Minute.json"
# series="usgs100_filtered_fixquery_e9_r100_idxlowe8"
#py -3.7 indexing.py \
#    --rebuild $sheets

#exit

# py -3.7 indexing.py \
#     --list $input \
#     $sheets

# mv index_result.csv icc_eval/index_result_$series"_actual.csv"

# exit
py -3.7 main.py \
    $input \
    $sheets \
    -r 100 \
    --noimg

py -3.7 eval_logs.py

mv eval_result.csv icc_eval/eval_results_$series"_actual.csv"
#!/bin/bash

input="/e/data/deutsches_reich/SBB/cut/list.txt"
sheets="data/blattschnitt_dr100_merged.geojson"

py -3.7 indexing.py \
    --list $input \
    $sheets \
#    --rebuild

exit
py -3.7 main.py \
    $input \
    $sheets \
    -r 20 \
    -v
py -3.7 eval_logs.py

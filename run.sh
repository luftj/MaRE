#!/bin/bash

py -3.7 download_maps.py
py -3.7 main.py data/input/list.txt data/blattschnitt_dr100_merged.geojson
py -3.7 eval_logs.py
py -3.7 eval_georef.py data/input/annotations_wiki.csv data/blattschnitt_dr100_merged.geojson
#!/bin/bash

# sheet name in $1

python main.py /e/data/deutsches_reich/SBB/cut/SBB_IIIC_Kart_L\ 1330_Blatt\ $1* data/blattschnitt_dr100_regular.geojson -r 1 --gt $1 -v #--rsize 1000
python eval_georef.py /e/data/deutsches_reich/SBB/cut/annotations.csv data/blattschnitt_dr100_regular.geojson --single $1
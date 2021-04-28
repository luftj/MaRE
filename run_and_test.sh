#!/bin/bash

# sheet name in $1
inputfile=$(ls /e/data/deutsches_reich/SBB/cut/SBB_IIIC_Kart_L\ 1330_Blatt\ $1*)
echo $inputfile
py -3.7 main.py "$inputfile" data/blattschnitt_dr100_regular.geojson -r 0 --gt $1 -v #--rsize 1000
py -3.7 eval_georef.py /e/data/deutsches_reich/SBB/cut/annotations.csv data/blattschnitt_dr100_regular.geojson --single $1
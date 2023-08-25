#!/bin/bash

# make sure, you got your environment activated!
# run "pip install -r requirements.txt" first to install all dependancies
# all "python" calls are python3

sheets_file="sampledata/blattschnitt_kdr100_fixed_dhdn.geojson"
input_list_file="sampledata/list.txt"

# get sample data
python sampledata/download_sample_data.py

# create index
if [[ -f "output/idx/index/index.ann" ]] 
then 
  echo "index already exists." 
else 
    python indexing.py $sheets_file --rebuild
fi

# optional: test index
# python indexing.py $sheets_file --list $input_list_file

# run georeferencing
python main.py $input_list_file $sheets_file -r 30

# evaluate logs to collect result data
python eval_logs.py

# collect some statistics for retrieval
python eval_scripts/eval_retrieval.py output/summary/eval_result.csv output/summary

# evaluate georeferencing accuracy
python eval_georef.py sampledata/annotations_wiki.csv $sheets_file --output output/summary/georef_stats.txt

# georeferenced output maps will be in ./output/
# result statistics will be in ./output/summary
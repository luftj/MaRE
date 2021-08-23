# MaRE

Extract water bodies from topographic maps and match them to OSM data for georeferencing.
---

This branch ([paper_supplement_icc2021](https://github.com/luftj/MaRE/tree/paper_supplement_icc2021)) contains the supplemental code to the paper "Content-based Image Retrieval for Map Georeferencing" published in Proceedings of the ICA 2021 (volume/issue tbd). This branch will be left stale.

We hope providing this repo helps other researchers to reproduce our method. If you want to expand on our method, please look at the current state on the master branch first. We are working on documenting the code better only on master. If you have any further questions, don't hesitate to contact us here via github or email.

## Installation

Requires
* Python3
* python-opencv

```$ python3 -m pip install -r requirements.txt ```


## Usage

Build index with:

`python3 indexing.py --rebuild [sheets geojson path]`

Get index ranks with:

`python3 indexing.py --list [file with list of query maps] [sheets geojson path]`

Georeference a single or all maps in a list with the following command:

`$ python3 main.py [image or list path] [sheets geojson path]`

The first run can take quite a while, because the OSM data needs to be downloaded and you will get rate limited. Please consider using your own overpass instance to avoid overloading the public servers.

## To do
* provide some sample data
* set config paths to work with sample data
* explain the data files and their layout a bit more
* check requirements.txt
* test installation and setup
* add some more description/documentation here :)
* add final paper PDF to this repo as soon as it is available and link it to this readme

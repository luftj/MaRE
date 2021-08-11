# MaRE

Extract water bodies from topographic maps and match them to OSM data for georeferencing.
---

This branch ([paper_supplement](https://github.com/luftj/MaRE/tree/paper_supplement)) contains the supplemental code to the paper "Automatic Content-Based Georeferencing of Historical Topographic Maps" published in Transactions in GIS (volume/issue tbd). It is a cleaned up version of [this](https://github.com/luftj/MaRE/releases/tag/cbgr-paper-revision) release. In the meantime a lot of advances have been made on master [repo](https://github.com/luftj/MaRE/tree/master), this branch will be left stale.

We hope providing this repo helps other researchers to reproduce our method. If you want to expand on our method, please look at the current state on the master branch first. We are working on documenting the code better only on master. If you have any further questions, don't hesitate to contact us here via github or email.

## Installation

Requires
* Python3
* python-opencv
* GDAL for evaluation

```$ python3 -m pip install -r requirements.txt ```


## Usage

First, you have to get some input data. In the paper we used most (some were added later and are not tested) maps on the wikipedia page of the [Karte des Deutschen Reiches 1:100000](https://de.wikipedia.org/wiki/Karte_des_Deutschen_Reiches_(Generalstabskarte)). Run the [script](download_maps.py) to download the maps and prepare all paths to reproduce our experiments.

Georeference a single or all maps in a list with the following command:

`$ python3 main.py [image or list path] [sheets geojson path]`

The first run can take quite a while, because the OSM data needs to be downloaded and you will get rate limited. Please consider using your own overpass instance to avoid overloading the public servers.

## data files

* input maps should be cropped to only contain the map artefact and no coloured background
* data/blattschnitt_*: these contain the sheet layout of the KDR100. *_regular contains the regular grid, *_merged has some grids merged to reflect overedge and irregularly sized sheets.
* todo

## To do
* set config paths to work with sample data
* explain the data files and their layout a bit more
* check requirements.txt
* add ground truth annotations
* make a run and eval script for a sample run
* test installation and setup
* add some more description/documentation here :)
* add final paper PDF to this repo as soon as it is available and link it to this readme

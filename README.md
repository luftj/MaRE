# MaRE

Extract water bodies from topographic maps and match them to OSM data for georeferencing.
---

This branch ([paper_supplement](https://github.com/luftj/MaRE/tree/paper_supplement)) contains the supplemental code to the paper XXX published at XXX. It is a cleaned up version of [this](https://github.com/luftj/MaRE/releases/tag/cbgr-paper-revision) release. In the meantime a lot of advances have been made on master [repo](https://github.com/luftj/MaRE/tree/master), this branch will be left stale.

We hope providing this repo helps other researchers to reproduce our method. If you want to expand on our method, please look at the current state on the master branch first. We are working on documenting the code better only on master. If you have any further questions, don't hesitate do contact us here via github or email.

## Installation

Requires
* Python3
* python-opencv

```$ python3 -m pip install -r requirements.txt ```


## Usage

First, you have to get some input data. In the paper we used all maps on the wikipedia page of the [Karte des Deutschen Reiches 1:100000](https://de.wikipedia.org/wiki/Karte_des_Deutschen_Reiches_(Generalstabskarte)). Run the prepare script (LINK) to download the maps and prepare all paths to reproduce our experiments.

`$ python3 main.py [image path] [sheets geojson path]`

## data files

* data/blattschnitt_*: these contain the sheet layout of the KDR100. *_regular contains the regular grid, *_merged has some grids merged to reflect overedge and irregularly sized sheets.
* todo

## To do
* prepare script (download from wikimedia)
* add link to prepare script to this readme
* set config paths to work with sample data
* explain the data files and their layout a bit more
* check requirements.txt
* test installation and setup
* add final paper PDF to this repo as soon as it is available and link it to this readme
# MaRE

Extract water bodies from topographic maps and match them to OSM data for georeferencing.
---

This branch ([paper_supplement](https://github.com/luftj/MaRE/tree/paper_supplement)) contains the supplemental code to the paper XXX published at XXX. It is a cleaned up version of [this](https://github.com/luftj/MaRE/releases/tag/cbgr-paper-revision) release. In the meantime a lot of advances have been made on master [repo](https://github.com/luftj/MaRE), this branch will be left stale.

We hope providing this repo helps other researchers to reproduce and expand on our method.

## Installation

Requires
* Python3
* python-opencv

```$ python3 -m pip install -r requirements.txt ```


## Usage

get some input data. In the paper we used all maps on the wikipedia page of the [Karte des Deutschen Reiches 1:100000](https://de.wikipedia.org/wiki/Karte_des_Deutschen_Reiches_(Generalstabskarte)). Run the prepare script (LINK) to download the maps and prepare all paths to reproduce our experiments.

`$ python3 main.py [image path] [sheets geojson path]`


# MaRE

Extract water bodies from topographic maps and match them to OSM data for georeferencing.
---

This branch (paper supplement) contains the supplemental code to the paper XXX published at XXX. It is a cleaned up version of [this](https://github.com/luftj/MaRE/releases/tag/cbgr-paper-revision) release. In the meantime a lot of advances have been made on master [repo](https://github.com/luftj/MaRE). 

We hope, providing this repo helps other researchers to reproduce and expand on our method.

## Installation

Requires
* Python3
* python-opencv

```$ python3 -m pip install -r requirements.txt ```


## Usage

`$ python3 main.py [image path] [sheets geojson path]`

## To Do
* collect and evaluate all parameters
* more descriptive output file names
* harmonise cv2/skimage usage where possible
* when not cropping map margins, remove as much of the nodata pixels as possible (least bbox)
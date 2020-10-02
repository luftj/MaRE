# MaRE

Extract water bodies from topographic maps and match them to OSM data for georeferncing.

WIP.

---

## Installation

Requires
* Python3
* python-opencv

```$ python3 -m pip install -r requirements.txt ```


## Usage

`$ python3 main.py [image path] [colour balance percent]`

## To Do
* automagically adjust segmentation parameters depending on image resolution
* collect and evaluate all parameters
* crop input image to remove scanning edge?
* log eval: avg time per sheet/target, number success, number correct
* move output to dir
* more descriptive output file names
* use original resolution map image for georeferencing
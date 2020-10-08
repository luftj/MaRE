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

`$ python3 main.py [image path] [sheets geojson path]`

## To Do
* collect and evaluate all parameters
* more descriptive output file names
* harmonise cv2/skimage usage where possible
* when not cropping map margins, remove as much of the nodata pixels as possible (least bbox)
* check if chunking image helps performance
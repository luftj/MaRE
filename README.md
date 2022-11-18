# MaRE

Automatically geoference map images by extracting symbols and aligning them with OpenStreetMap.

---

## Introduction

This software allows you to georeference map sheets of historical topographic map series. The basic working principle is to use segmented map symbols for content-based image retrieval. The reference maps that will be retrieved and thus serve as ground control for georeferencing are fetched from OpenStreetMap.

Theory, application and evaluation has been treated in the following pulbications:
* 05/2023: *Automatische Georeferenzierung von Altkarten*, PhD Thesis. Coming soon.
* 09/2022: *Georeferenzierung automatisieren?* Workshop presentation at the FID-Karten, Berlin (link tba)
* 08/2021: *Automatic Content-Based Georeferencing of Historical Topographic Maps*, journal paper, Transactions in GIS [PDF](https://onlinelibrary.wiley.com/doi/10.1111/tgis.12794)
* 12/2021: *Content-based Image Retrieval for Map Georeferencing*, conference paper, International Cartographic Conference 2021 [PDF](http://jonasluft.de/data/ICC21_full_paper_submission.pdf)

## Licensing and distribution

This project is part of my PhD Thesis at the [g2lab](http://www.geomatik-hamburg.de/g2lab/) of the HCU Hamburg and was partially funded by the City of Hamburg via the [sharing.city.college](https://ahoi.digital/sharing-city-college/) of [ahoi.digital](https://ahoi.digital/).

The repo contains unlicensed third-party code [here](simple_cb.py). I have no rights for the sample data contained in XXX, their copyright is most likely expired.

If you would like to use parts or all of this code, please get in touch.
## Installation

Requires
* Python3 (tested with python 3.7 and 3.10)
* GDAL binaries

```$ python3 -m pip install -r requirements.txt ```

OpenCV has changed the bindings for the non-free feature descriptors a couple of times in previous versions, so there might be some issues (tested with opencv-contrib-python==4.6.0.66).

## Configuration
Outputh paths, OpenStreetMap mirror and processing parameters are set in the [configuration file](config.py). The most important parameters are:

* to do


## Usage

The most simple usage is as follows:
`$ python3 main.py [input] [sheets]`
This will georeference the image contained at __input__. At the path __sheets__ there should be a GeoJSON file, containing polygon features describing all possible map locations' bounding boxes (make sure there are only 5 points per geometry). See the [sample data](...) as an example.

If __input__ is not an image, it should be a text file with a line-delimited list of image paths and ground truths to be georeferenced. See the [sample data](...) for formatting.

The image(s) given with __input__ can either be 3- to 4-channel colour images, which will then be segmented. If thei are 1-channel images, they are assumed to already be segmentation masks.

The number of hypotheses to spatially verify during retrievel can be set with the parameter __-r__. Use a low number to save computation time. Use a higher number to reduce the likelihood of wrongfully discarding the correct location hypotheses (and thus increase possible prediction accuracy).

You can automatically crop map margins in the output images by providing the __--crop__ parameter.

To process the sample data run the following:
`$ python3 main.py example/list.txt example/sheets.geojson`

## Future work
There are a couple of venues, how the performance of this system can be improved. I will probably not get to doing it myself in the near future. If you would like to work in this, give me a shout :)

* find feature descriptors that better work with high-contrast/low-texture segmentation masks
* find a dense registration method that is less susceptible to local minima and bad initialisation
* use transforms that better fit expected distortions in map artefacts
* loosen the requirements for the map sheet prior knwowledge
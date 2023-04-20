# MaRE

Automatically geoference map images by extracting symbols and aligning them with OpenStreetMap.

---

## Introduction

This software allows you to georeference map sheets of historical topographic map series. The basic working principle is to use segmented map symbols for content-based image retrieval. The reference maps that will be retrieved and thus serve as ground control for georeferencing are fetched from OpenStreetMap.

Theory, application and evaluation has been treated in the following pulbications:
* 05/2023: *Automatische Georeferenzierung von Altkarten*, PhD thesis. Coming soon.
* 09/2022: *Georeferenzierung automatisieren?* Workshop presentation at the FID-Karten, Berlin [slides](https://kartdok.staatsbibliothek-berlin.de/receive/kartdok_mods_00000486)
* 08/2021: *Automatic Content-Based Georeferencing of Historical Topographic Maps*, journal paper, Transactions in GIS [full text](https://onlinelibrary.wiley.com/doi/10.1111/tgis.12794)
* 12/2021: *Content-based Image Retrieval for Map Georeferencing*, conference paper, International Cartographic Conference 2021 [full text](https://doi.org/10.5194/ica-proc-4-69-2021)

## Licensing and distribution

This project is part of my PhD thesis at the [g2lab](http://www.geomatik-hamburg.de/g2lab/) of the HCU Hamburg and was partially funded by the City of Hamburg via the [sharing.city.college](https://ahoi.digital/sharing-city-college/) of [ahoi.digital](https://ahoi.digital/).

The repo contains unlicensed third-party code [here](simple_cb.py). I have no rights for the sample data, their copyright is most likely expired -- downlaod at your own discretion.

If you would like to use parts or all of this code, please get in touch.
## Installation

Requires
* Python3 (tested with python 3.7 and 3.10)
* GDAL binaries

```$ python3 -m pip install -r requirements.txt ```

OpenCV has changed the bindings for the non-free feature descriptors a couple of times in previous versions, so there might be some issues (tested with opencv-contrib-python==4.6.0.66).

## Configuration
Outputh paths, OpenStreetMap mirror and processing parameters are set in the [configuration file](config.py). Look there for allowed values. The most important parameters are:

* base_path -- where you want your output
* base_path_index -- in case you are reusing the index for multiple experiments, set this to be independant from base_path
* proj_map -- proj4 string that defines the projection/SRS of your input maps
* proj_sheets -- the projection of your sheets/quadrangle file in case it is different from the maps
* sheet_name_field -- the property in the sheets/quadrangle definition file that contains the unique names/ids for every sheet
* osm_url -- when you get rate limited by overpass when downloading OSM data, set up your own instance and put the link here
* osm_query -- this is the overpass query that is used to get the relevant reference data. This needs to be adjusted depending on the symbols you extract from the input maps
* draw_ocean_polygon -- set this to true, if you want to draw the ocean as a (filled) polygon. Otherwise, only coastlines will be drawn as lines
* segmentation_steps -- this determines the segmentation pipeline of your input maps. For possible values refer to segmentation.py
* early_termination_heuristic -- set this to True if you want to save some georeferencing time. But you might get some more incorrect results
* skip_impossible_verification -- save some time on georeferencing, when it is not possible to get a correct result. Only works when supplying ground truth
* warp_mode_retrieval/warp_mode_registration -- set the transform class for retrieval/registration to fit to the expected deformation in your input maps.
* proj_out -- the desired projection of the georeferenced output maps

## Usage

The most simple usage is as follows:
`$ python3 main.py [input] [sheets]`
This will georeference the image contained at __input__. At the path __sheets__ there should be a GeoJSON file, containing polygon features describing all possible map sheet/quadrangle locations' bounding boxes (make sure there are only 5 points per geometry). See the [sample data](sampledata/blattschnitt_kdr100_fixed_dhdn.geojson) as an example.

If __input__ is not an image, it should be a text file with a line-delimited list of image paths and ground truths to be georeferenced. See the [sample data](sampledata/list.txt) for formatting.

The image(s) given with __input__ can either be 3- to 4-channel colour images, which will then be segmented. If thei are 1-channel images, they are assumed to already be segmentation masks.

The number of hypotheses to spatially verify during retrieval can be set with the parameter __-r__. Use a low number to save computation time. Use a higher number to reduce the likelihood of wrongfully discarding the correct location hypotheses (and thus increase possible prediction accuracy). 30 was a sensible number for most experiments.

You can automatically crop map margins in the output images by providing the __--crop__ parameter.

## Run with sample data
To process the sample data do the following:
1. create python environment with `$ python3 -m venv env`
2. activate environment with `$ source env/bin/activate` (on linux and bash)
3. install required binary dependancies (see above)
4. install required python dependancies with `$ pip install -r requirements.txt`
5. download and process sample data `$ bash process_sample_data.sh`. The first time this might take a while to download all reference data.
6. georeferenced maps will appear in ./output/
7. statistics of the results will appear in ./output/summary

## Notes

* there are some helper scripts in eval_scripts to analyse result files. There are some hard-coded bits in there you might want to change before using them
* the scripts that where used for all evaluation during writing of the thesis are in experiments/. You will have to change paths etc. in there to reproduce them
* there are some drawio figures in docs/ that were used in the dissertation. When you want to use them, please cite the dissertation (citation info will follow after publication)

## How to cite this work

to do -- citation info will follow after publication of the dissertation. 

For now you can cite the [ICC paper](https://doi.org/10.5194/ica-proc-4-69-2021).

## Future work
There are a couple of venues, how the performance of this system can be improved. I will probably not get to doing it myself in the near future. If you would like to work on this, give me a shout :)

* find feature descriptors that better work with high-contrast/low-texture segmentation masks
* find a dense registration method that is less susceptible to local minima and bad initialisation
* use transforms that better fit expected distortions in map artefacts
* loosen the requirements for the map sheet prior knwowledge
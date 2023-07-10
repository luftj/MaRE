# web GUI for map georeferencer

## intro

A user can upload a map image, set parameters and start the georeferencing process. THe georeferencing is then done by a different app and once it is done, the result will be displayed to the user.


## overview

### pages

1. index: upload map & set configs
2. result: view and download the resulting georeference
3. segmentation: tune the segmentation params (this will be added in a later version)

### handling state

User will get a session, which keeps an ID, all config and the name of the uploaded image. The image will be saved in DATA_DIR.

## installation & setup

### env vars

* DATA_DIR - where to save the input map image and the result (too big for memory). Will be mounted as volume in prod

### dev environment

1. install requirements with `pip -r requirements.txt`
2. run `flask --app gui_app.main run`
3. open `localhost:5000` in your browser

### prod environment

To do: provide Dockerfile

## to do

* prod deployment
* get possible indexes from file
* make config param list smarter: disable if from index, add info text, ...
* call georeferencing backend
* outfile should have filename including input filename
* allow to set output path for georeferencing
* implement way of prvoding config file to georef method
* allow umlaute in output filename
* get error messages from backend
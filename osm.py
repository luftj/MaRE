import json
import os
import overpass
import cv2
import numpy as np
from time import sleep

def get_from_osm(bbox=[16.3,54.25,16.834,54.5]):
    data_path = "data/osm/rivers_%s.geojson" % "_".join(map(str,bbox))

    # don't query if we already have the data on disk
    if os.path.isfile( data_path ):
        with open(data_path, encoding="utf-8") as file:
            json_data = json.load(file)
            return json_data

    sorted_bbox = ",".join(map(str,[bbox[1], bbox[0], bbox[3], bbox[2]]))
    query = '(way (%s) [water=lake]; way (%s) [natural=water] [name]; way (%s) [type=waterway] [name]; way (%s) [waterway=river] [name];);out geom;' % (sorted_bbox,sorted_bbox,sorted_bbox,sorted_bbox) # ; (._;>;)
    # print(query)

    api = overpass.API()
    while True:
        try:
            result = api.get(query)
            break
        except Exception as e:
            print("overpass query failed!",e)
            sleep(10)
            continue # try again
    
    # print("#raw features:",len(result["features"]))
    # clean up geojson: no empty geometries
    result["features"] = [ f for f in result["features"] if len(f["geometry"]["coordinates"]) > 0]
    # filter geojson: no point features
    result["features"] = [ f for f in result["features"] if f["geometry"]["type"] != "Point"]
    for feat in result["features"]:
        if feat["geometry"]["coordinates"][0] == feat["geometry"]["coordinates"][-1]:
            feat["geometry"]["type"] = "Polygon"
            feat["geometry"]["coordinates"] = [feat["geometry"]["coordinates"]]
    # print("#line+poly features:",len(result["features"]))
    
    with open(data_path, mode="w", encoding="utf-8") as f:
        json.dump(result, f)

    return result

def coord_to_point(coords, bbox, img_size, castint=True):
    lon = coords[0]
    lat = coords[1]
    x = (lon-bbox[0]) / (bbox[2]-bbox[0]) * img_size[0]
    y = (lat-bbox[1]) / (bbox[3]-bbox[1]) * img_size[1]
    y = img_size[1]-y
    if castint:
        x = int(x)
        y = int(y)
    return (x,y)

def paint_features(json_data, bbox=[16.3333,54.25,16.8333333,54.5], img_size=(1000,850)):
    image = np.zeros(shape=img_size[::-1], dtype=np.uint8)
    # print(image.shape)
    for feature in json_data["features"]:
        if feature["geometry"]["type"] == "LineString":
            points = [ coord_to_point(p,bbox,img_size) for p in feature["geometry"]["coordinates"] ]
            thickness = 2 if ("waterway" in feature["properties"] and feature["properties"]["waterway"] == "river") else 1
            for idx in range(len(points)-1):
                cv2.line(image, points[idx], points[idx+1], 255, thickness=thickness)
        elif feature["geometry"]["type"] == "Polygon":
            points = [ coord_to_point(p,bbox,img_size) for p in feature["geometry"]["coordinates"][0] ]
            points = np.array(points)
            # cv2.fillConvexPoly(image, points, 255)
            cv2.fillPoly(image, [points], 255)
        else:
            raise NotImplementedError("drawing feature type not implemented %s!" % feature["geometry"]["type"])
    return image
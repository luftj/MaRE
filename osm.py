import json
import os
import cv2
import numpy as np
from time import sleep
import requests
import logging
from osmtogeojson import osmtogeojson
from pyproj import Transformer

from config import path_osm, proj_map, proj_osm, proj_sheets

transform_osm_to_map = Transformer.from_proj(proj_osm, proj_map, skip_equivalent=True, always_xy=True)
transform_sheet_to_osm = Transformer.from_proj(proj_sheets, proj_osm, skip_equivalent=True, always_xy=True)
transform_sheet_to_map = Transformer.from_proj(proj_sheets, proj_map, skip_equivalent=True, always_xy=True)

def query_overpass(query):
    import overpass
    api = overpass.API()
    while True:
        try:
            result = api.get(query, responseformat="json")
            break
        except Exception as e:
            print("overpass query failed!",e)
            sleep(10)
            continue # try again
    
    result = as_geojson(result["elements"])

    print("#raw features:",len(result["features"]))
    # clean up geojson: no empty geometries
    result["features"] = [ f for f in result["features"] if len(f["geometry"]["coordinates"]) > 0]
    # filter geojson: no point features
    result["features"] = [ f for f in result["features"] if f["geometry"]["type"] != "Point"]
    for feat in result["features"]:
        if "type" in feat["properties"] and feat["properties"]["type"] == "multipolygon" and feat["geometry"]["coordinates"][0][0] is list:# and (len(feat["geometry"]["coordinates"][0]) > 2 or len(feat["geometry"]["coordinates"][0]) == 1):
            print("MP, len coord",len(feat["geometry"]["coordinates"][0]))
            feat["geometry"]["type"] = "MultiPolygon"
        elif feat["geometry"]["coordinates"][0] == feat["geometry"]["coordinates"][-1]:
            feat["geometry"]["type"] = "Polygon"
            feat["geometry"]["coordinates"] = [feat["geometry"]["coordinates"]]
    print("#line+poly features:",len(result["features"]))
    return result

def get_from_osm(bbox=[16.3,54.25,16.834,54.5], url = #"https://overpass.openstreetmap.ru/api/interpreter"):
    #"https://overpass.osm.ch/api/interpreter"):
    "http://overpass-api.de/api/interpreter"):
    data_path = path_osm + "rivers_%s.geojson" % "_".join(map(str,bbox))

    minxy = transform_sheet_to_osm.transform(bbox[0], bbox[1]) # reproject lower left bbox corner
    maxxy = transform_sheet_to_osm.transform(bbox[2], bbox[3]) # reproject upper right bbox corner
    bbox = minxy+maxxy

    # don't query if we already have the data on disk
    if os.path.isfile( data_path ):
        logging.debug("fetching osm data from disk: %s" % data_path)
        with open(data_path, encoding="utf-8") as file:
            json_data = json.load(file)
            return json_data

    
                # way (%s) [water=river];
                # way (%s) [waterway=riverbank];
                # way (%s) [waterway=ditch];
                # way (%s) [waterway=drain];

    sorted_bbox = ",".join(map(str,[bbox[1], bbox[0], bbox[3], bbox[2]]))
    query = """[out:json];
                (nwr (%s) [water=lake]; 
                way (%s) [natural=water] [name]; 
                way (%s) [type=waterway] [name]; 
                way (%s) [waterway=river] [name];
                way (%s) [waterway=canal] [name];
                way (%s) [water=river];
                way (%s) [waterway=stream] [name];
                way (%s) [natural=coastline];
                );
                out body;
                >;
                out skel qt;""" % ((sorted_bbox,)*8) # ; (._;>;)
    logging.debug("osm query: %s" % query)

    while True:
        try:
            result = requests.get(url, params={'data': query})
            result = result.json()
            break
        except Exception as e:
            import re
            error_msg = re.findall("error[^<]*",result.text)
            if len(error_msg) == 0:
                logging.critical(result.text)
                print(result.text)
                raise(e)
            logging.error(error_msg)
            if "rate_limited" in error_msg[0] or "timeout" in error_msg[0]:
                logging.warning("timeout or rate limited, retrying in 5 sec...")
                sleep(5)
                continue
            else:
                print("unknown error" + result.text)
                logging.critical("unknown error" + result.text)
                raise(e)
    gj = osmtogeojson.process_osm_json(result)
    
    with open(data_path, mode="w", encoding="utf-8") as f:
        json.dump(gj, f)

    return gj

def coord_to_point(coords, bbox, img_size, castint=True):
    coords = transform_osm_to_map.transform(coords[0], coords[1])
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
    minxy = transform_sheet_to_map.transform(bbox[0], bbox[1]) # reproject lower left bbox corner
    maxxy = transform_sheet_to_map.transform(bbox[2], bbox[3]) # reproject upper right bbox corner
    bbox = minxy+maxxy

    image = np.zeros(shape=img_size[::-1], dtype=np.uint8)
    for feature in json_data["features"]:
        try:
            if feature["geometry"]["type"] == "LineString":
                points = [ coord_to_point(p,bbox,img_size) for p in feature["geometry"]["coordinates"] ]
                if "waterway" in feature["properties"] and feature["properties"]["waterway"] == "river":
                    thickness = 2  
                elif "natural" in feature["properties"] and feature["properties"]["natural"] == "coastline":
                    thickness = 5
                else:
                    thickness = 1
                for idx in range(len(points)-1):
                    cv2.line(image, points[idx], points[idx+1], 255, thickness=thickness)
            elif feature["geometry"]["type"] == "Polygon":
                points = [ coord_to_point(p,bbox,img_size) for p in feature["geometry"]["coordinates"][0] ]
                points = np.array(points)
                # cv2.fillConvexPoly(image, points, 255)
                cv2.fillPoly(image, [points], 255)
            elif feature["geometry"]["type"] == "MultiPolygon":
                for poly in feature["geometry"]["coordinates"][0]:
                    points = [ coord_to_point(p,bbox,img_size) for p in poly ]
                    points = np.array(points)
                    # cv2.fillConvexPoly(image, points, 255)
                    cv2.fillPoly(image, [points], 255)
            else:
                raise NotImplementedError("drawing feature type not implemented %s!" % feature["geometry"]["type"])
        except Exception as e:
            logging.error(e)
            errortext = "Error parsing feature at "+ ",".join(map(str,bbox)) + json.dumps(feature["properties"]) + feature["geometry"]["type"]
            logging.error(errortext)
    return image

if __name__ == "__main__":
    # download data
    # todo: load balance
    import find_sheet
    import progressbar
    import sys
    # create necessary directories
    import os
    os.makedirs("logs/", exist_ok=True)
    os.makedirs(path_osm, exist_ok=True)

    logging.basicConfig(filename='logs/osm.log', level=logging.DEBUG) # gimme all your loggin'!
    progress = progressbar.ProgressBar()
    sheets_file = "data/blattschnitt_dr100_regular.geojson"
    bboxes = find_sheet.get_bboxes_from_json(sheets_file)
    # bboxes = bboxes[250:]
    if len(sys.argv) == 1:
        for bbox in progress(bboxes):
            gj = get_from_osm(bbox)
        exit()
    
    bbox = bboxes[find_sheet.get_index_of_sheet(sheets_file, sys.argv[1])]
    gj = get_from_osm(bbox)
    img = paint_features(gj,bbox)
    from config import jpg_compression
    cv2.imwrite("outref.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpg_compression])

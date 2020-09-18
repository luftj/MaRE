import json
import os
import overpass
from time import sleep


def get_from_osm(bbox=[16.3,54.25,16.834,54.5]):
    data_path = "data/osm/rivers_%s.geojson" % "_".join(map(str,bbox))

    # don't query if we already have the data on disk
    if os.path.isdir( data_path ):
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
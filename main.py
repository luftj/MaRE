import cv2
import argparse
import numpy as np
import overpass
import json
from matplotlib import pyplot as plt

from simple_cb import simplest_cb

def extract_blue(img):
    img_cb = simplest_cb(img, args.percent)
    img_cie = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    img_cie = simplest_cb(img_cie, args.percent)

    # TODO: adjust kernel sizes to image resolution
    ksize = (5, 5) 
    img_cie = cv2.blur(img_cie, ksize)  

    l,a,b = cv2.split(img_cie)

    # cv2.imshow("b",cv2.resize(b,(b.shape[1]//2,b.shape[0]//2)))

    lowerBound = (10, 0, 0)
    upperBound = (255, 90, 70)

    img_thresh = cv2.inRange(img_cie, lowerBound, upperBound)
    # cv.Not(cv_rgb_thresh, cv_rgb_thresh)

    # retm,b_threshold = cv2.threshold(b,5,255,cv2.THRESH_BINARY)
    # cv2.imshow("b_threshold",cv2.resize(b_threshold,(b_threshold.shape[1]//2,b_threshold.shape[0]//2)))
    # retm,a_threshold = cv2.threshold(a,5,255,cv2.THRESH_BINARY)
    # cv2.imshow("a_threshold",cv2.resize(a_threshold,(a_threshold.shape[1]//2,a_threshold.shape[0]//2)))

    # plt.subplot(2,2,1), plt.imshow(img)
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,2), plt.imshow(img_cie)
    # plt.title('LAB Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,3), plt.imshow(img_thresh)
    # plt.title('thresh Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,4), plt.hist((b.ravel(),a.ravel()), 256)
    # plt.title('b Histogram'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # ksize = (5, 5) 
    # opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, ksize)
    # # cv2.imshow("opening",cv2.resize(opening,(opening.shape[1]//2,opening.shape[0]//2)))

    # ksize = (33, 33) 
    # # closing = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, ksize)
    # # cv2.imshow("closing",cv2.resize(closing,(closing.shape[1]//2,closing.shape[0]//2)))

    # dilation = cv2.dilate(img_thresh,ksize,iterations = 4)
    # cv2.imshow("dilation",cv2.resize(dilation,(dilation.shape[1]//2,dilation.shape[0]//2)))

    return img_thresh

def get_from_osm(bbox=[16.3,54.25,16.834,54.5]):
    sorted_bbox = ",".join(map(str,[bbox[1],bbox[0],bbox[3],bbox[2]]))
    query = '(way (%s) [water=lake]; way (%s) [natural=water] [name]; way (%s) [type=waterway] [name]; way (%s) [waterway=river] [name];);out geom;' % (sorted_bbox,sorted_bbox,sorted_bbox,sorted_bbox) # ; (._;>;)

    print(query)
    api = overpass.API()
    result = api.get(query)
    
    print("#raw features:",len(result["features"]))
    # clean up geojson: no empty geometries
    result["features"] = [ f for f in result["features"] if len(f["geometry"]["coordinates"]) > 0]
    # filter geojson: no point features
    result["features"] = [ f for f in result["features"] if f["geometry"]["type"] != "Point"]
    for feat in result["features"]:
        if feat["geometry"]["coordinates"][0] == feat["geometry"]["coordinates"][-1]:
            feat["geometry"]["type"] = "Polygon"
            feat["geometry"]["coordinates"] = [feat["geometry"]["coordinates"]]
    print("#line+poly features:",len(result["features"]))
    
    with open("rivers_%s.geojson" % "_".join(map(str,bbox)),mode="w") as f:
        json.dump(result,f)

    return result

def coord_to_point(coords, bbox, img_size, castint=True):
    # todo: reproject bbox and coords to UTM, to avoid geodetic distortion

    lon = coords[0]
    lat = coords[1]
    x = (lon-bbox[0])/(bbox[2]-bbox[0])*img_size[0]
    y = (lat-bbox[1])/(bbox[3]-bbox[1])*img_size[1]
    y = img_size[1]-y
    if castint:
        x = int(x)
        y = int(y)
    return (x,y)

def paint_rivers(json_data, bbox=[16.3,54.25,16.834,54.5], img_size=(1000,1000)):
    image = np.zeros(shape=img_size, dtype=np.uint8)
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

def which_utm_zone(bbox):
    if bbox[2] < bbox[0]:
        raise ValueError("maxx < minx! bbox in wrong order or across antemeridian")

    lon,lat = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2) # WARNING: won't work when crossing the antemeridian!
    num = (lon + 180) // 6 + 1
    n_s = "N" if lat > 0 else "S"
    zone = "%d%s" % (num, n_s)
    return zone

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path string")
    # parser.add_argument("output", help="output file path string")
    parser.add_argument("percent", help="threshold", default=5,type=int)
    args = parser.parse_args()

    bbox = [16.3,54.25,16.834,54.5]
    print("UTM zone:", which_utm_zone(bbox))

    rivers_json = get_from_osm(bbox)
    reference_river_image = paint_rivers(rivers_json, bbox)

    img = cv2.imread(args.input)
    cv2.imshow("img", cv2.resize(img,(img.shape[1]//4,img.shape[0]//4)))

    water_mask = extract_blue(img)
    cv2.imshow("water mask from map", cv2.resize(water_mask,(water_mask.shape[1]//4,water_mask.shape[0]//4)))

    # cv2.imwrite(args.output, out)

    cv2.imshow("reference rivers from OSM", cv2.resize(reference_river_image,(reference_river_image.shape[1]//2,reference_river_image.shape[0]//2)))
    cv2.waitKey()
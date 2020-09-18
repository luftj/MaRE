import cv2
import argparse
import numpy as np
import json
import os

import find_sheet
import osm
import segmentation
from matching import feature_matching_brief

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

def compute_similarities(query_image, reference_image):
    # from skimage.metrics import structural_similarity as ssim
    # from skimage.metrics import mean_squared_error
    
    if cv2.countNonZero(query_image) == 0 or cv2.countNonZero(reference_image) == 0:
        # empty image
        return [0]
    
    query_image_resized = cv2.resize(query_image,reference_image.shape[::-1])
    n_matches = feature_matching_brief(query_image_resized,reference_image)
    
    # mse = mean_squared_error(query_image_resized, reference_image)
    # ssim_v = ssim(query_image_resized, reference_image, data_range=reference_image.max() - reference_image.min())
    return [n_matches]

def retrieve_best_match(query_image, bboxes):
    closest_image = None
    closest_bbox = None
    best_dist = -1

    import time
    start_time = time.time()

    for idx,bbox in enumerate(bboxes):
        time_now = time.time()
        rivers_json = osm.get_from_osm(bbox)
        reference_river_image = paint_features(rivers_json, bbox)
        
        # todo: detect border in query image
        # reference_river_image = cv2.copyMakeBorder(reference_river_image, 50,150,50,50, cv2.BORDER_CONSTANT, None, 0)

        distances = compute_similarities(water_mask, reference_river_image)

        if closest_image is None or distances[0] > best_dist:
            closest_image = reference_river_image
            closest_bbox = bbox
            best_dist = distances[0]

        print("%d/%d" % (idx, len(bboxes)),"Distances:", *distances, bbox, time.time()-time_now)
        time_now = time.time()

    end_time = time.time()
    print("time spent:", end_time - start_time)

    return closest_image,closest_bbox,best_dist

def georeference(inputfile, outputfile, bbox):

    left, top, right, bottom = (bbox[0], bbox[3], bbox[2], bbox[1])

    command = "gdal_translate -of GTiff -a_ullr %f %f %f %f -a_srs EPSG:4269 %s %s" % (left, top, right, bottom, inputfile, outputfile)
    print(command)
    os.system(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path string")
    # parser.add_argument("output", help="output file path string")
    parser.add_argument("percent", help="threshold", default=5, type=int)
    args = parser.parse_args()

    sheets_file = "data/blattschnitt_dr100.geojson"
    bboxes = find_sheet.get_bboxes_from_json(sheets_file)
    bboxes = bboxes[624:627]

    img = cv2.imread(args.input)

    water_mask = segmentation.extract_blue(img, args.percent)

    # find the bbox for this query image
    closest_image, closest_bbox, dist = retrieve_best_match(water_mask, bboxes)
    
    sheet_name = find_sheet.find_name_for_bbox(sheets_file, closest_bbox)
    print("best sheet:", sheet_name, "with", dist)

    # cv2.imshow("img", cv2.resize(img, (500,500)))
    # cv2.imshow("water mask from map", cv2.resize(water_mask, (500,500)))
    # cv2.imshow("closest reference rivers from OSM", cv2.resize(closest_image, (500,500)))
    # cv2.waitKey()

    cv2.imwrite("refimg_%s_%s.jpg" % (sheet_name, "-".join(map(str,closest_bbox))), closest_image)

    # todo: register query and retrieved reference image for fine alignment
    import registration
    query_image_small = cv2.resize(water_mask,(500,500))
    closest_image_border = cv2.copyMakeBorder(closest_image, 150,150,150,150, cv2.BORDER_CONSTANT, None, 0)
    reference_image_small = cv2.resize(closest_image_border,(500,500))
    warp_matrix = registration.register_ECC(query_image_small,reference_image_small)
    # crop out border
    # aligned_query = aligned_query[border_x:aligned_query.shape[1]-border_x, border_y:aligned_query.shape[0]-border_y]
    # cv2.imshow("aligned query", aligned_query)
    # cv2.imwrite("aligned_%s_%s.jpg" % (sheet_name, "-".join(map(str,closest_bbox))), closest_image)
    # query_scale_x = water_mask.shape[1] / query_image_small.shape[1]
    # query_scale_y = water_mask.shape[0] / query_image_small.shape[0]
    query_aligned = registration.warp(query_image_small,warp_matrix)
    orig_small = cv2.resize(img,(500,500))
    im_aligned = registration.warp(orig_small,warp_matrix)
    border_x = int(150 * reference_image_small.shape[1] / closest_image_border.shape[1])
    border_y = int(150 * reference_image_small.shape[0] / closest_image_border.shape[0])
    print(closest_image.shape,reference_image_small.shape,border_x,border_y)
    im_aligned = im_aligned[border_y:im_aligned.shape[0]-border_y, border_x:im_aligned.shape[1]-border_x]
    # im_aligned = cv2.warpAffine(orig_small, warp_matrix, (500,500,3), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    cv2.imshow("query aligned", query_aligned)
    cv2.imshow("aligned orig", im_aligned)
    aligned_path = "aligned_%s_%s.jpg" % (sheet_name, "-".join(map(str,closest_bbox)))
    cv2.imwrite(aligned_path, im_aligned)

    # georeference aligned query image with bounding box
    georeference(aligned_path, "georef_sheet_%s.tif" % sheet_name, closest_bbox)

    cv2.waitKey(0)
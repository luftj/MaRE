import cv2
import argparse
import numpy as np
import json
from matplotlib import pyplot as plt

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

def calculate_HOG(image):
    import matplotlib.pyplot as plt

    from skimage.feature import hog
    from skimage import exposure

    image = cv2.resize(image,(1000,1000))

    image = np.array(image,dtype=np.float)
    image /= 255

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, feature_vector=False)
    nz = np.where(fd != 0)
    
    print(len(fd),len(nz[0]),nz)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    return fd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path string")
    # parser.add_argument("output", help="output file path string")
    parser.add_argument("percent", help="threshold", default=5, type=int)
    args = parser.parse_args()

    bboxes = find_sheet.get_bboxes_from_json("data/blattschnitt_dr100.geojson")
    bboxes = bboxes[610:750]

    img = cv2.imread(args.input)

    water_mask = segmentation.extract_blue(img, args.percent)

    closest_image = None
    closest_bbox = None
    min_dist = -1

    import time
    start_time = time.time()

    for bbox in bboxes:
        time_now = time.time()
        rivers_json = osm.get_from_osm(bbox)
        reference_river_image = paint_features(rivers_json, bbox)
        
        reference_river_image = cv2.copyMakeBorder(reference_river_image,50,150,50,50,cv2.BORDER_CONSTANT,None,0)

        distances = compute_similarities(water_mask, reference_river_image)

        if closest_image is None or distances[0] > min_dist:
            closest_image= reference_river_image
            closest_bbox = bbox
            min_dist = distances[0]

        print("Distances:",*distances, bbox,time.time()-time_now)
        time_now = time.time()

    end_time = time.time()
    print("time spent:",end_time - start_time)
    print("best sheet:", find_sheet.find_name_for_bbox("data/blattschnitt_dr100.geojson",closest_bbox),"with",min_dist)

    cv2.imshow("img", cv2.resize(img,(img.shape[1]//4,img.shape[0]//4)))
    cv2.imshow("water mask from map", cv2.resize(water_mask,(water_mask.shape[1]//4,water_mask.shape[0]//4)))
    cv2.imshow("closest reference rivers from OSM", cv2.resize(closest_image,(closest_image.shape[1]//2,closest_image.shape[0]//2)))
    cv2.waitKey()

    # cv2.imwrite(args.output, out)

    cv2.waitKey(20)
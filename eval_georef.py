import csv
import argparse
import json
import re
import os
import profile
import time

import pyproj
from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread
from skimage import data
from skimage.feature import match_template, peak_local_max
from skimage.transform import resize
from PIL import Image

from find_sheet import find_poly_for_name

from config import path_output, proj_sheets, proj_out

def match_sheet_name(img_name):
    # s = re.findall(r"(?<=_)[0-9][0-9][0-9a](?=_)",img_name)
    # s = re.findall(r"[0-9][0-9][0-9a](?=_)",img_name)
    # s = re.findall(r"(?<=[\s_])*[0-9]?[0-9][0-9a](?=[_\s])",img_name)
    s = re.findall(r"(?<=[\s_])[0-9]?[0-9][0-9a](?=[_\s])",img_name)
    s = [e.lstrip('0') for e in s]
    sheet_name = "-".join(s)
    return sheet_name

def read_corner_CSV(filepath):
    print(filepath)

    sheet_corners = {}

    with open(filepath, encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=',', quotechar='"')
        for row in reader:
            img_name = row["#filename"]
            corner_data = json.loads(row["region_shape_attributes"])
            if corner_data == {}:
                continue
            point = (corner_data["cx"], corner_data["cy"])
            

            # print(img_name, sheet_name, point)

            if not img_name in sheet_corners:
                sheet_corners[img_name] = []
            sheet_corners[img_name].append(point)
    
    sheet_corners = { s:p for (s,p) in sheet_corners.items() if len(p) == 4}
    return sheet_corners

def match_corner(image, template):
    result = match_template(image, template,pad_input=True)
    # plt.imshow(result)
    # plt.show()
    result[result > 1] = 0 # there are some weird artefacts in the nodata area!
    # peaks = peak_local_max(result, min_distance=min(template.shape)//2)
    # scores = [result[y,x] for (y,x) in peaks]
    # scores.sort(reverse=True)
    # print(scores)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    corr_coef = result[y,x]
    return (x, y)

def get_coords_from_raster(georef_image_path, point):
    import rasterio # leave this in here, to avoid proj_db errors
    dataset = rasterio.open(georef_image_path)

    latlong = dataset.transform * point

    return latlong

def mean_absolute_error(points_a, points_b):
    points_a = np.array(points_a).T
    points_b = np.array(points_b).T

    xx = points_a.reshape(2, -1)
    yy = points_b.reshape(2, -1)

    geod = pyproj.Geod(ellps='WGS84')
    _, _, distances = geod.inv(
            xx[0,:],
            xx[1,:],
            yy[0,:],
            yy[1,:])

    # distances = np.hypot(*(xx - yy))
    sum_errors = np.sum(distances)
    
    return sum_errors/4

def root_mean_squared_error(points_a, points_b):
    points_a = np.array(points_a).T
    points_b = np.array(points_b).T

    xx = points_a.reshape(2, -1)
    yy = points_b.reshape(2, -1)

    geod = pyproj.Geod(ellps='WGS84')
    _, _, distances = geod.inv(
            xx[0,:],
            xx[1,:],
            yy[0,:],
            yy[1,:])
    print("distances",distances)
    sq_distances = np.square(distances)
    print("squared distances",sq_distances)

    # distances = np.hypot(*(xx - yy))
    sum_errors = np.sum(sq_distances)
    
    return np.sqrt(sum_errors/4)

def warp_images(filenames,inputpath):
    for img_name in filenames:
        print("warping", img_name)
        sheet_name = match_sheet_name(img_name)

        img_path = inputpath + "/" + img_name
        im = Image.open(img_path)
        width, height = im.size

        georef_path = path_output + "georef_sheet_%s_warp.tif" % sheet_name
        command = "gdalwarp -order 1 -t_srs EPSG:4326 -ts %d %d -overwrite %s/georef_sheet_%s.jp2 %s" %(width, height, path_output, sheet_name, georef_path)
        print("exec: %s" % command)
        os.system(command)

def get_truth_bbox(sheets, sheet_name):
    transform_sheet_to_out = pyproj.Transformer.from_proj(proj_sheets, proj_out, skip_equivalent=True, always_xy=True)

    truth_bbox = find_poly_for_name(sheets, sheet_name)
    
    if len(truth_bbox) != 5:
        raise ValueError("bbox should have 4 points, has %d: %s" % (len(truth_bbox), truth_bbox))

    truth_bbox = [transform_sheet_to_out.transform(x, y) for (x,y) in truth_bbox]
    return truth_bbox

def findCorners(img, georef_img, ref_corners, plot=False, template_size = 20):
    corner_points = []
    for idx,point in enumerate(ref_corners):
        x_min = max(0,point[1]-template_size)
        x_max = min(img.shape[0],point[1]+template_size)
        y_min = max(0,point[0]-template_size)
        y_max = min(img.shape[1],point[0]+template_size)
        template = img[x_min:x_max, y_min:y_max]
        match = match_corner(georef_img, template)
        corner_points.append(match)

        if plot:
            # show corners
            ax = plt.subplot(2, 4, idx+1)
            ax.set_title("original")
            plt.xticks([],[])
            plt.yticks([],[])
            plt.gray()
            plt.imshow(template)
            ax = plt.subplot(2, 4, idx+5)
            ax.set_title("warped")
            plt.xticks([],[])
            plt.yticks([],[])
            plt.gray()
            x_min = max(0,match[1]-template_size)
            x_max = min(georef_img.shape[0],match[1]+template_size)
            y_min = max(0,match[0]-template_size)
            y_max = min(georef_img.shape[1],match[0]+template_size)
            plt.imshow(georef_img[x_min:x_max, y_min:y_max])
            # TODO: plot center point, to show pixel perfect location
    if plot:
        plt.show()
    return corner_points

def cascadeCorners(img_path, georef_path, truth_corners, plot, downscale_factor):
    img = imread(img_path, as_gray=True)
    georef_img = imread(georef_path, as_gray=True)

    # downscale images
    small_width = img.shape[1]//downscale_factor
    img_small = resize(img, (int(small_width/img.shape[1]*img.shape[0]), small_width), anti_aliasing=True)
    georef_img_small = resize(georef_img, (int(small_width/georef_img.shape[1]*georef_img.shape[0]),small_width), anti_aliasing=True)
    # rescale truth corners to small resolution
    scaled_corners = [ (x * img_small.shape[1]//img.shape[1], y  * img_small.shape[0]//img.shape[0]) for (x,y) in truth_corners]
    # find  corners in small image
    corner_points = findCorners(img_small, georef_img_small, scaled_corners, template_size=20, plot=args.plot)
    # rescale found coordinates to original resolution
    corner_points = [ (x * georef_img.shape[1]//georef_img_small.shape[1], y  * georef_img.shape[0]//georef_img_small.shape[0]) for (x,y) in corner_points]
    corner_coords = []
    for idx,corner in enumerate(corner_points):
        roi_size = 100
        template_size = 20
        # extract ROI from original size image
        x_min = max(0,corner[1]-roi_size)
        x_max = max(0,corner[1]+roi_size)
        y_min = max(0,corner[0]-roi_size)
        y_max = max(0,corner[0]+roi_size)
        roi = georef_img[x_min:x_max,y_min:y_max]
        ref_corner = truth_corners[idx]
        template = img[ref_corner[1]-template_size:ref_corner[1]+template_size, ref_corner[0]-template_size:ref_corner[0]+template_size]
        # match again in ROIs
        match = match_corner(roi, template)
        match = (match[0]+(y_min), match[1]+(x_min)) # scale match to non-ROI positions
        corner_coords.append(get_coords_from_raster(georef_path, match))

        if plot:
            # show corners
            ax = plt.subplot(2, 4, idx+1)
            ax.set_title("original")
            plt.xticks([],[])
            plt.yticks([],[])
            plt.gray()
            plt.imshow(template)
            ax = plt.subplot(2, 4, idx+5)
            ax.set_title("warped")
            plt.xticks([],[])
            plt.yticks([],[])
            plt.gray()
            x_min = max(0,match[1]-template_size)
            x_max = min(georef_img.shape[0],match[1]+template_size)
            y_min = max(0,match[0]-template_size)
            y_max = min(georef_img.shape[1],match[0]+template_size)
            plt.imshow(georef_img[x_min:x_max, y_min:y_max])
            # plt.imshow(georef_img[match[1]-template_size:match[1]+template_size, match[0]-template_size:match[0]+template_size])
            # TODO: plot center point, to show pixel perfect location
    if plot:
        plt.show()

    return corner_coords

def dump_csv(sheets_list, mae_list, rmse_list):
    print("writing to file...")
    with open("eval_georef_result.csv", "w", encoding="utf-8") as eval_fp:
        eval_fp.write("sheet name; MAE [m]; RMSE [m]\n") # header

        for sheet, mae, rmse in zip(sheets_list, mae_list, rmse_list):
            eval_fp.write("%s; %.2f; %.2f\n" % (sheet, mae, rmse))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path string with corner annotations")
    parser.add_argument("sheets", help="sheets json file path string", default="data/blattschnitt_dr100.geojson")
    parser.add_argument("--plot", help="set this to true to show debugging plots", action="store_true")
    parser.add_argument("--nowarp", help="set this to not update warped images", action="store_true")
    parser.add_argument("--single", help="provide sheet number to test only a single sheet", default=None)
    parser.add_argument("--ransac", help="...", action="store_true")
    args = parser.parse_args()
    # python eval_georef.py /e/data/deutsches_reich/wiki/highres/382.csv data/blattschnitt_dr100_merged_digi.geojson
    
    inputpath = os.path.dirname(args.input)

    sheet_corners = read_corner_CSV(args.input)
    img_list = list(sheet_corners.keys())#[-5:-4]

    if args.single:
        img_list = [x for x in img_list if match_sheet_name(x)==args.single]

    if not args.nowarp:
        warp_images(img_list,inputpath) # this has to be done before calculating coords, because proj db breaks

    error_results = []
    rmse_results = []
    sheet_names = []

    for img_name in img_list:
        t0 = time.time()
        print(img_name)
        sheet_name = match_sheet_name(img_name)

        img_path = inputpath + "/" + img_name
        georef_path = path_output + "/georef_sheet_%s_warp.tif" % sheet_name
        
        if not os.path.exists(georef_path):
            print("Couldn't find file for registered sheet %s.\nIt probably failed to get a registration solution" % sheet_name)
            continue

        # find corner coordinates of registered image (geo-coordinates)
        corner_coords = cascadeCorners(img_path, georef_path, sheet_corners[img_name], plot=args.plot, downscale_factor=4)

        truth_bbox = get_truth_bbox(args.sheets, sheet_name)
        mae = mean_absolute_error(corner_coords[0:4], truth_bbox[0:4])
        rmse = root_mean_squared_error(corner_coords[0:4], truth_bbox[0:4])
        # if mae > 2000:
        #     print("mean absolute error: %f m" % mae)
        #     print("root mean squared error: %f m" % rmse)
        #     # maybe a corner wasn't properly detected, try again with full resolution ;)
        #     corner_coords = cascadeCorners(img_path, georef_path, sheet_corners[img_name], plot=args.plot, downscale_factor=1)
        #     truth_bbox = get_truth_bbox(args.sheets, sheet_name)
        #     mae = mean_absolute_error(corner_coords[0:4], truth_bbox[0:4])
        #     rmse = root_mean_squared_error(corner_coords[0:4], truth_bbox[0:4])
        print("mean absolute error: %f m" % mae)
        print("root mean squared error: %f m" % rmse)
        error_results.append(mae)
        rmse_results.append(rmse)
        sheet_names.append(sheet_name)

        print("time for image:", time.time() - t0, "s")

        if args.plot:
            plt.show()
    
    total_mean_error = sum(error_results)/len(error_results)
    total_mean_rmse = sum(rmse_results)/len(rmse_results)
    print("total mean error: %f m" % total_mean_error)
    print("total mean RMSE: %f m" % total_mean_rmse)

    results_sorted = sorted(zip(sheet_names,error_results), key=lambda tup: tup[1])
    sheet_names_sorted = [x[0] for x in results_sorted]
    error_sorted = [x[1] for x in results_sorted]

    median_error = error_sorted[len(error_sorted)//2]
    print("median MAE: %f m" % median_error)

    results_sorted = sorted(zip(sheet_names,rmse_results), key=lambda tup: tup[1])
    sheet_names_sorted = [x[0] for x in results_sorted]
    error_sorted = [x[1] for x in results_sorted]

    median_error = error_sorted[len(error_sorted)//2]
    print("median RMSE: %f m" % median_error)

    if not args.single:
        dump_csv(sheet_names, error_results, rmse_results)

    plt.subplot(2, 1, 1)
    plt.bar(sheet_names_sorted, error_sorted)
    plt.axhline(total_mean_error, c="g", linestyle="--", label="mean")
    plt.annotate("%.0f" % total_mean_error,(0,total_mean_error + 100))
    plt.axhline(median_error, c="r", label="median")
    plt.annotate("%.0f" % median_error,(0,median_error + 100))
    plt.legend()
    plt.title("average error per sheet [m]")
    plt.subplot(2, 1, 2)
    plt.title('error distribution total [m]')
    plt.boxplot(error_sorted, vert=False, showmeans=True, medianprops={"color":"r"})
    plt.axhline(total_mean_error, xmax=0, c="g", label="mean")
    plt.axhline(median_error, xmax=0, c="r", label="median")
    plt.legend()
    plt.show()

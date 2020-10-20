import csv
import argparse
import json
import re
import os
import time

import pyproj
from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread
from skimage import data
from skimage.feature import match_template
from PIL import Image
import dask_image.imread
import dask.array
import dask.delayed
import skimage.util

from find_sheet import find_poly_for_name
from config import path_output

def match_sheet_name(img_name):
    # s = re.findall(r"(?<=_)[0-9][0-9][0-9a](?=_)",img_name)
    s = re.findall(r"[0-9][0-9][0-9a](?=_)",img_name)
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
    # result = skimage.util.apply_parallel(match_template, image,
    #                                 dtype=np.float64, chunks=None, #dtype requires skimage>=0.18
    #                                 extra_keywords={"template":template,"pad_input":True})
    
    result = match_template(image, template, pad_input=True)
    # print(result)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    corr_coef = result[y,x]
    return (x, y)

def get_coords_from_raster(georef_image_path, point):
    import rasterio # leave this in here, to avoid proj_db errors
    dataset = rasterio.open(georef_image_path)

    latlong = dataset.transform * point

    return latlong

def mean_squared_error(points_a, points_b):
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

    # distances = np.hypot(*(xx - yy))
    sum_errors = np.sum(distances)
    
    return sum_errors/4

def warp_images(filenames,inputpath):
    for img_name in filenames:
        print("warping", img_name)
        sheet_name = match_sheet_name(img_name)

        img_path = inputpath + "/" + img_name
        im = Image.open(img_path)
        width, height = im.size

        georef_path = path_output + "georef_sheet_%s_warp.tif" % sheet_name
        command = "gdalwarp -t_srs EPSG:4326 -ts %d %d -overwrite %s/georef_sheet_%s.jp2 %s" %(width, height, path_output, sheet_name, georef_path)
        print("exec: %s" % command)
        os.system(command)

def grayscale(rgb):
    result = ((rgb[..., 0] * 0.2125) +
              (rgb[..., 1] * 0.7154) +
              (rgb[..., 2] * 0.0721))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path string with corner annotations")
    parser.add_argument("sheets", help="sheets json file path string", default="data/blattschnitt_dr100.geojson")
    parser.add_argument("--plot", help="set this to true to show debugging plots", action="store_true")
    parser.add_argument("--nowarp", help="set this to not update warped images", action="store_true")
    args = parser.parse_args()
    # python eval_georef.py /e/data/deutsches_reich/wiki/highres/382.csv data/blattschnitt_dr100_merged_digi.geojson
    t0=time.time()    
    inputpath = os.path.dirname(args.input)

    sheet_corners = read_corner_CSV(args.input)
    img_names = list(sheet_corners.keys())[9:10]
    t1=time.time()
    print("time parse csv", t1-t0)
    if not args.nowarp:
        warp_images(img_names,inputpath) # this has to be done before calculating coords, because proj db breaks
    print("time warp", time.time()-t1)
    error_results = []
    sheet_names = []

    template_size = 20

    times = []

    for img_name in img_names:
        time_start = time.time()
        print(img_name)
        sheet_name = match_sheet_name(img_name)
        truth_bbox = find_poly_for_name(args.sheets, sheet_name)

        img_path = inputpath + "/" + img_name
        img = dask.delayed(imread)(img_path, as_gray=True)
        
        # img = dask_image.imread.imread(img_path)
        # img = grayscale(img)
        # img = img[0, ...]
        # img = img.rechunk("auto")
        # print(img)

        georef_path = path_output + "/georef_sheet_%s_warp.tif" % sheet_name
        georef_img = dask.delayed(imread)(georef_path, as_gray=True)

        # georef_img = dask_image.imread.imread(georef_path)
        # georef_img = grayscale(georef_img)
        # georef_img = georef_img[0, ...]
        # georef_img = georef_img.rechunk("auto")
        # print(georef_img)

        corner_coords = []

        for idx,point in enumerate(sheet_corners[img_name]):
            y = point[1]
            x = point[0]
            template = img[y-template_size:y+template_size, x-template_size:x+template_size]
            match = dask.delayed(match_corner)(georef_img, template)
            coords = get_coords_from_raster(georef_path, match)
            corner_coords.append(coords)
            # print(point, match, coords, truth_bbox[idx])

            if args.plot:
                # show corners
                plt.subplot(2, 4, (idx*2)+1)
                plt.gray()
                plt.imshow(template)
                plt.subplot(2, 4, (idx*2)+2)
                plt.gray()
                plt.imshow(georef_img[match[1]-template_size:match[1]+template_size, match[0]-template_size:match[0]+template_size])

        # corner_coords = dask.compute(*corner_coords)
        print("\n\nresult", corner_coords, truth_bbox[0:4])
        mse = mean_squared_error(corner_coords, truth_bbox[0:4]) # TODO: corners should be sorted, can't rely on order in sheets or annotation
        print("mean error: %f m" % mse)
        error_results.append(mse)
        sheet_names.append(sheet_name)
        times.append(time.time() - time_start)
        print("time taken: %f" % times[-1])
        if args.plot:
            plt.show()
    
    total_mean_error = sum(error_results)/len(error_results)
    total_mean_error = total_mean_error.compute()
    print("total mean error: %f m" % total_mean_error)
    
    avg_time = sum(times)/len(times)
    print("avg eval time per sheet: %f s" % avg_time)
    print("total time",time.time()-t0)

    plt.bar(sheet_names, error_results)
    plt.title("average error per sheet")
    plt.show()

    



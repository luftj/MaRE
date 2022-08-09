import csv
import argparse
import json
import re
import os
import sys
# import profile
import time
import glob
from operator import itemgetter

import pyproj
from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.feature import match_template, peak_local_max
from skimage.transform import downscale_local_mean, rescale
from skimage import filters

from find_sheet import find_poly_for_name, get_poly_dict
import config

def match_sheet_name(img_name):
    # s = re.findall(r"(?<=_)[0-9][0-9][0-9a](?=_)",img_name)
    # s = re.findall(r"[0-9][0-9][0-9a](?=_)",img_name)
    # s = re.findall(r"(?<=[\s_])*[0-9]?[0-9][0-9a](?=[_\s])",img_name) # also matches invventory key in SBB set
    s = re.findall(r"(^[0-9]{3}(?=[\.]))",img_name) # SLUB z.b. "002.bmp"
    s += re.findall(r"((?<=[\s_])[0-9]?[0-9][0-9a](?=[_\s]))",img_name) # SBB z.b. "SBB_IIIC_Kart_L 1330_Blatt 258 von 1925_koloriert.tif"
    s += re.findall(r"^[0-9]{3}(?=\_)",img_name) # DK50 z.B. "344_Guben_1937.png"
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
            
            if not img_name in sheet_corners:
                sheet_corners[img_name] = []
            sheet_corners[img_name].append(point)
        
    # sort corners to be: top-left, top-right, bottom-right, bottom-left
    for sheet_name, points in sheet_corners.items():
        sorted_by_y = sorted(points, key=itemgetter(1))
        top = sorted(sorted_by_y[0:2], key=itemgetter(0))
        bot = sorted(sorted_by_y[-2:], key=itemgetter(0), reverse=True)
        sorted_points = top + bot
        sheet_corners[sheet_name] = sorted_points
    
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

    # distances = np.hypot(*(xx - yy))
    sum_errors = np.sum(sq_distances)
    
    return np.sqrt(sum_errors/4)

def get_truth_bbox(sheets, sheet_name):
    transform_sheet_to_out = pyproj.Transformer.from_proj(config.proj_sheets, config.proj_out, skip_equivalent=True, always_xy=True)

    truth_bbox = find_poly_for_name(sheets, sheet_name)
    
    if len(truth_bbox) != 5:
        raise ValueError("bbox should have 4 points, has %d: %s" % (len(truth_bbox), truth_bbox))

    truth_bbox = [transform_sheet_to_out.transform(x, y) for (x,y) in truth_bbox]
    return truth_bbox

def get_truth_bbox_dict(sheets_dict, sheet_name):
    transform_sheet_to_out = pyproj.Transformer.from_proj(config.proj_sheets, config.proj_out, skip_equivalent=True, always_xy=True)

    truth_bbox = sheets_dict[sheet_name]
    
    if len(truth_bbox) != 5:
        raise ValueError("bbox should have 4 points, has %d: %s" % (len(truth_bbox), truth_bbox))

    truth_bbox = [transform_sheet_to_out.transform(x, y) for (x,y) in truth_bbox]
    return truth_bbox

def findCorners(img, georef_img, ref_corners, plot=False, template_size = 20):
    corner_points = []
    # g_w = georef_img.shape[1]//2
    # g_h = georef_img.shape[0]//2
    # quarter_images = [  ((  0,  0), georef_img[0:g_h,0:g_w]), # top left
    #                     ((g_w,  0), georef_img[0:g_h,g_w: ]), # top right
    #                     ((g_w,g_h), georef_img[g_h: ,g_w: ]), # bot right
    #                     ((  0,g_h), georef_img[g_h: ,0:g_w])] # bot left
    # pre quart: average time per image: 10.690692
    # post quart: average time per image: 10.442163
    # plt.subplot(2,2,1)
    # plt.imshow(quarter_images[0])
    # plt.subplot(2,2,2)
    # plt.imshow(quarter_images[1])
    # plt.subplot(2,2,3)
    # plt.imshow(quarter_images[2])
    # plt.subplot(2,2,4)
    # plt.imshow(quarter_images[3])
    # plt.show()
    # exit()
    for idx,point in enumerate(ref_corners):
        x_min = max(0,point[1]-template_size)
        x_max = min(img.shape[0],point[1]+template_size)
        y_min = max(0,point[0]-template_size)
        y_max = min(img.shape[1],point[0]+template_size)
        template = img[x_min:x_max, y_min:y_max]
        match = match_corner(georef_img, template)
        # offset, quart = quarter_images[idx]
        # match = match_corner(quart, template)
        # match = (match[0] + offset[0], match[1] + offset[1])
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
    print(img_path,georef_path)
    img = imread(img_path, as_gray=True)
    georef_img = imread(georef_path, as_gray=True)
    scale = georef_img.shape[1] / img.shape[1] # if image has been resized during georeferencing, templates might not match anymore
    if scale != 1:
        img = rescale(img, scale, anti_aliasing=False)
        # but now the transform/annotations doesn't fit anymore... rescale truth corners
    if downscale_factor > 1:
        # downscale images
        img_small = downscale_local_mean(img, (downscale_factor, downscale_factor))
        georef_img_small = downscale_local_mean(georef_img, (downscale_factor, downscale_factor))
        # blurring adds 0.8s (6.7->7.5) per image on average, but helps with correct corner identification (even though 3,3 gaussian is a bit much...)
        img_small = filters.gaussian(img_small, sigma=(3, 3), truncate=1.0)
        georef_img_small = filters.gaussian(georef_img_small, sigma=(3, 3), truncate=1.0)
        
        # rescale truth corners to small resolution
        scaled_corners = [ (int(x*scale / downscale_factor), int(y*scale / downscale_factor)) for (x,y) in truth_corners]
        # find  corners in small image
        corner_points = findCorners(img_small, georef_img_small, scaled_corners, template_size=20, plot=plot)
        # rescale found coordinates to original resolution
        corner_points = [ (x * downscale_factor, y  * downscale_factor) for (x,y) in corner_points]
    else:
        scaled_corners = [ (int(x*scale), int(y*scale)) for (x,y) in truth_corners]
        corner_points = findCorners(img, georef_img, scaled_corners, template_size=50, plot=plot)
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
        ref_corner = [ int(c*scale) for c in ref_corner]
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

def dump_csv(sheets_list, mae_list, rmse_list, outpath="eval_georef_result.csv", append=False):
    print("writing to file...")
    with open(outpath, "a" if append else "w", encoding="utf-8") as eval_fp:
        if not append:
            eval_fp.write("sheet name; MAE [m]; RMSE [m]\n") # header

        for sheet, mae, rmse in zip(sheets_list, mae_list, rmse_list):
            eval_fp.write("%s; %.2f; %.2f\n" % (sheet, mae, rmse))

def eval_list(img_list, sheet_corners, inputpath, sheetsfile, images_path, plot=False, downscale_factor=6, outfile=None ):
    error_results = []
    rmse_results = []
    sheet_names = []
    times = []

    truth_bboxes = get_poly_dict(sheetsfile)

    for img_name in img_list:
        t0 = time.time()
        print(img_name)
        sheet_name = match_sheet_name(img_name)

        img_path = inputpath + "/" + img_name
        georef_path = images_path + "/aligned_%s_*" % sheet_name
        georef_path_glob = glob.glob(georef_path)
        georef_path_glob = [x for x in georef_path_glob if x[-4:] != ".wld"]
        
        if len(georef_path_glob) == 0:
            print("Couldn't find file for registered sheet %s.\nIt probably failed to get a registration solution" % sheet_name)
            continue
        georef_path = georef_path_glob[0]

        # find corner coordinates of registered image (geo-coordinates)
        corner_coords = cascadeCorners(img_path, georef_path, sheet_corners[img_name], plot=plot, downscale_factor=downscale_factor)

        truth_bbox = get_truth_bbox_dict(truth_bboxes, sheet_name)
        mae = mean_absolute_error(corner_coords[0:4], truth_bbox[0:4])
        rmse = root_mean_squared_error(corner_coords[0:4], truth_bbox[0:4])
        
        print("mean absolute error: %f m" % mae)
        print("root mean squared error: %f m" % rmse)
        
        error_results.append(mae)
        rmse_results.append(rmse)
        sheet_names.append(sheet_name)
        if outfile:
            with open(outfile,"a") as fw:
                fw.write("%s; %.2f; %.2f\n" % (sheet_name, mae, rmse))

        time_taken = time.time() - t0
        times.append(time_taken)
        print("time for image:", time_taken, "s")

        if plot:
            plt.show()

    if len(times)>0:
        print("average time per image: %f" % (sum(times)/len(times)))

    return sheet_names, error_results, rmse_results

def summary_and_fig(annotations_file, sheets_file, single=False, outfile=sys.stdout, debug_plot=False, append_to=None, downscale_factor=6 ):
    inputpath = os.path.dirname(annotations_file)

    sheet_corners = read_corner_CSV(annotations_file)
    img_list = list(sheet_corners.keys())#[-5:-4]
    
    if single:
        img_list = [x for x in img_list if match_sheet_name(x)==single]
    elif append_to:
        # part of the dataset was already evaluated, only do the missing sheets
        done_sheets = []
        with open(append_to, "r") as fr:
            fr.readline() # header
            for line in fr:
                sheet_name, _, _ = line.split("; ")
                done_sheets.append(sheet_name.zfill(3))
        img_list = list(filter(lambda x: x.split(".")[0] not in done_sheets, img_list))

    outpath=(append_to if append_to else config.path_output+"/eval_georef_result.csv")
    with open(outpath, "a" if append_to else "w", encoding="utf-8") as eval_fp:
        if not append_to:
            eval_fp.write("sheet name; MAE [m]; RMSE [m]\n") # header
    sheet_names, error_results, rmse_results = eval_list(
                                                    img_list, 
                                                    sheet_corners, 
                                                    inputpath, 
                                                    sheets_file, 
                                                    config.path_output, 
                                                    debug_plot, 
                                                    downscale_factor=downscale_factor,
                                                    outfile=outpath)
    
    sheet_names = []
    error_results = []
    rmse_results = []
    with open(outpath) as fr:
        fr.readline()
        for line in fr:
            sheet, mae, rmse = line.strip().split("; ")
            sheet_names.append(sheet)
            error_results.append(float(mae))
            rmse_results.append(float(rmse))

    if len(error_results) == 0:
        return
    total_mean_error = sum(error_results)/len(error_results)
    total_mean_rmse = sum(rmse_results)/len(rmse_results)
    print("total mean error: %f m" % total_mean_error, file=outfile)
    print("total mean RMSE: %f m" % total_mean_rmse, file=outfile)

    results_sorted = sorted(zip(sheet_names,error_results), key=lambda tup: tup[1])
    sheet_names_sorted = [x[0] for x in results_sorted]
    error_sorted = [x[1] for x in results_sorted]

    median_error_mae = error_sorted[len(error_sorted)//2]
    print("median MAE: %f m" % median_error_mae, file=outfile)

    results_sorted = sorted(zip(sheet_names,rmse_results), key=lambda tup: tup[1])
    sheet_names_sorted = [x[0] for x in results_sorted]
    error_sorted = [x[1] for x in results_sorted]

    median_error_rmse = error_sorted[len(error_sorted)//2]
    print("median RMSE: %f m" % median_error_rmse, file=outfile)

    # if not single:
    #     dump_csv(sheet_names, error_results, rmse_results, outpath=(append_to if append_to else "eval_georef_result.csv"), append=append_to)

    plt.subplot(2, 1, 1)
    plt.bar(sheet_names_sorted, error_sorted)
    plt.axhline(total_mean_rmse, c="g", linestyle="--", label="mean")
    plt.annotate("%.0f" % total_mean_rmse,(0,total_mean_rmse + 100))
    plt.axhline(median_error_rmse, c="r", label="median")
    plt.annotate("%.0f" % median_error_rmse,(0,median_error_rmse + 100))
    plt.legend()
    plt.title("average error per sheet [m]")
    plt.subplot(2, 1, 2)
    plt.title('error distribution total [m]')
    plt.boxplot(error_sorted, vert=False, showmeans=True, medianprops={"color":"r"})
    plt.axhline(total_mean_rmse, xmax=0, c="g", label="mean")
    plt.axhline(median_error_rmse, xmax=0, c="r", label="median")
    plt.legend()
    if not debug_plot:
        plt.savefig("georef_error.png")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path string with corner annotations")
    parser.add_argument("sheets", help="sheets json file path string", default="data/blattschnitt_dr100.geojson")
    parser.add_argument("--plot", help="set this to true to show debugging plots", action="store_true")
    parser.add_argument("--single", help="provide sheet number to test only a single sheet", default=None)
    parser.add_argument("--output", help="store result figures and summary here", type=str, default=None)
    args = parser.parse_args()
    # python eval_georef.py /e/data/deutsches_reich/wiki/highres/382.csv data/blattschnitt_dr100_merged_digi.geojson
    # py -3.7 -m cProfile -s "cumulative" eval_georef.py /e/data/deutsches_reich/wiki/highres/annotations_wiki.csv data/blattschnitt_dr100_merged.geojson > profile2.txt
    
    summary_and_fig(args.input,args.sheets, single=args.single, outfile=args.output, debug_plot=args.plot, downscale_factor=6)
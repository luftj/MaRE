# exp 8a: find out, wether hypetheses about reasons for bad registration quality are correct, by registering subwindows of sampel maps

# sample zones:
# A good 389, 413
# A bad 388, 414
# B good 554, 568
# B bad 553, 569
# C good 623, 636
# C bad 622, 637

import experiments.config_e8 as config
import os
import glob
from eval_scripts.eval_helpers import load_errors_csv
import cv2
import numpy as np
from matplotlib import pyplot as plt

import segmentation
from registration import register_ECC, warp
from main import scale_proportional
from find_sheet import find_bbox_for_name
import osm

def get_eccs(resultfile):
    result = {}
    with open(resultfile) as fr:
        fr.readline() # skip header
        for line in fr:
            line = line.strip().split("; ")
            sheet = line[0]
            ecc = line[-1]
            result[sheet] = ecc
    return result

def get_query_image(images_list, sheet_query):
    path = os.path.dirname(images_list)+ "/"
    with open(images_list) as fr:
        for line in fr:
            filename,sheet = line.strip().split(",")
            if sheet == sheet_query:
                path += filename
                return path

def get_reference_image(sheet):
    sheetfile = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
    closest_bbox = find_bbox_for_name(sheetfile, sheet)
    rivers_json = osm.get_from_osm(closest_bbox)
    closest_image = osm.paint_features(rivers_json, closest_bbox)
    return closest_image

images_list = "E:/data/deutsches_reich/SLUB/cut/raw/list.txt"
exp_dir = "E:/experiments/e8"#_oldseg/"
reg_resultsfile = f"{exp_dir}/eval_georef_result.csv"
loc_resultsfile = f"{exp_dir}/eval_result.csv"
out_dir = "E:/experiments/e8a"
os.makedirs(out_dir, exist_ok=True)

errors = load_errors_csv(reg_resultsfile)
eccs = get_eccs(loc_resultsfile)
sheets_of_interest = ["388","414","553","569","622","637"]

for sheet in sheets_of_interest:
    print("sheet:",sheet)
    print("MAE:",errors[sheet])
    print("ECC:",eccs[sheet])

    # get input file
    query_image_path = get_query_image(images_list, sheet)
    print(query_image_path)
    query_img = cv2.imdecode(np.fromfile(query_image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # make segmentation of input file
    query_mask = segmentation.extract_blue(query_img)

    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    # downscale to precessing size
    processing_size = scale_proportional(query_img.shape, config.process_image_width)

    query_img_small = cv2.resize(query_img, processing_size, config.resizing_register_query)
    query_mask_small = cv2.resize(query_mask, processing_size, config.resizing_register_query)
    
    reference_image = get_reference_image(sheet)
    border_size = config.template_window_size
    reference_image = cv2.resize(reference_image, 
                                    (processing_size[0] - border_size*2,
                                     processing_size[1] - border_size*2),
                                    config.resizing_register_reference)
    reference_image = cv2.copyMakeBorder(reference_image, 
                                                border_size, border_size, border_size, border_size, 
                                                cv2.BORDER_CONSTANT, None, 0)

    x_parts = 3
    y_parts = 3
    q_height, q_width = query_mask_small.shape
    r_height, r_width = reference_image.shape
    print(q_width,q_height)
    print(r_width,r_height)
    # for each part
    index = 1
    for y in range(y_parts):
        for x in range(x_parts):
            print(x,y)
            # cut query image + segmentation mask in parts
            query_img_small_part = query_img_small[
                                    y*q_height//y_parts:(y+1)*q_height//y_parts,
                                    x*q_width//x_parts:(x+1)*q_width//x_parts]
            query_mask_small_part = query_mask_small[
                                    y*q_height//y_parts:(y+1)*q_height//y_parts,
                                    x*q_width//x_parts:(x+1)*q_width//x_parts]
            # cut reference image
            reference_img_part = reference_image[
                                    y*r_height//y_parts:(y+1)*r_height//y_parts,
                                    x*r_width//x_parts:(x+1)*r_width//x_parts]
            
            plt.subplot(y_parts*2,x_parts*3,index)
            plt.imshow(query_img_small_part)
            plt.subplot(y_parts*2,x_parts*3,index+x_parts)
            plt.imshow(query_mask_small_part)

            # run register_ECC
            try:
                warp_matrix, ecc = register_ECC(query_mask_small_part,reference_img_part, ret_cc=True)
                map_img_aligned = warp(query_img_small_part, warp_matrix)
                plt.subplot(y_parts*2,x_parts*3,index+x_parts*3*y_parts)
                plt.imshow(map_img_aligned)
            except:
                print("not converged")

            plt.subplot(y_parts*2,x_parts*3,index+x_parts*3*y_parts+x_parts)
            plt.imshow(reference_img_part)
            plt.title(f"{float(ecc):.3f}")
            index += 1
        index += x_parts * 2
    
    # get registered image from exp8
    aligned_image_path = glob.glob(exp_dir+"/aligned_"+sheet+"_*")[0]
    print(aligned_image_path)
    aligned_image = cv2.imdecode(np.fromfile(aligned_image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
    # visualise all parts next to each other vs whole image registered
    plt.subplot(2,3,3)
    plt.imshow(query_img)
    plt.subplot(2,3,6)
    plt.imshow(aligned_image)
    # check ECC value
    plt.title(f"{float(eccs[sheet]):.3f}")
    plt.show()
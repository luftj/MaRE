from email.mime import base
import os,glob
import numpy as np
from PIL import Image

from find_sheet import find_poly_for_name, get_poly_dict
from eval_georef import get_coords_from_raster, get_truth_bbox_dict, mean_absolute_error

def distance(a, b, border=0):
    return abs(np.linalg.norm(a-b)-border)

def calc_mae_px(query_corners,border_corners,transform):
    mae_px = 0
    for q,b in zip(query_corners, border_corners):
        warped = transform @ q
        dist = distance(warped,b)
        mae_px += dist
        # print("px",b,warped,dist)
    mae_px /= 4
    return mae_px

def calc_mae_m(query_corners,transform,image_path,truth_bbox):
    warped_corners = []
    for p in query_corners:
        warped = transform @ p
        # proj corner points to wgs84
        warped_coords = get_coords_from_raster(image_path,warped[0:2])
        # print("m",p,warped,warped_coords)
        warped_corners.append(warped_coords)
    mae_m = mean_absolute_error(warped_corners[0:4], truth_bbox[0:4])  # calc geodesic distance
    return mae_m

def dump_csv(scores_file,scores):
    # write to file
    with open(scores_file,"w") as fw:
        fw.write("sheet;mae px;mae m\n")
        for sheet in scores:
            fw.write(f"{sheet};{scores[sheet]['mae px']};{scores[sheet]['mae m']}\n")

def calc_all_scores(sheetsfile, out_dir, baseline_margin = 100):
    # get ground truth corner points from sheets file
    truth_bboxes = get_poly_dict(sheetsfile)

    scores = {}
    for file in glob.glob(os.path.join(out_dir,"transform_*.npy")):#[10:11]:
        sheet_name = file.split("_")[-1].split(".")[0]
        transform = np.load(file) # get transform for image
        transform = np.vstack([transform,[0,0,1]]) # homogeneous transform

        # get corner points
        border = np.load(out_dir+"border_"+sheet_name+".npy")
        
        # adjust for border
        image_path = glob.glob(os.path.join(out_dir,f"aligned_{sheet_name}_*.bmp"))[0]
        im = Image.open(image_path)
        img_width, img_height = im.size # mind the non-square sheets 10 and 72
        baseline_margin = int(100 * img_width/1200)
        scale_mat = np.eye(3,3,dtype=np.float32)
        # get scaleing from iamge size and border+margin
        # scale_x = (img_width-2*reference_border)/(img_width-2*baseline_margin)
        # scale_y = (img_height-2*reference_border)/(img_height-2*baseline_margin)
        # scale_x = (img_width-2*border[0])/(img_width-2*baseline_margin)
        # scale_y = (img_height-2*border[0])/(img_height-2*baseline_margin)
        scale_mat[0,0] = (img_width-border[0])/(img_width-baseline_margin)# 1.025  # x scaling factor
        scale_mat[1,1] = (img_height-border[0])/(img_height-baseline_margin)# 1.029 # y scaling factor
        transform = scale_mat @ np.linalg.inv(transform) @ np.linalg.inv(scale_mat)

        query_corners = [
            np.asarray([baseline_margin,
                        baseline_margin,
                        1]),  # upper_left
            np.asarray([img_width - baseline_margin,
                        baseline_margin,
                        1]), # upper_right
            np.asarray([img_width - baseline_margin,
                        img_height - baseline_margin,
                        1]), # lower_right
            np.asarray([baseline_margin,
                        img_height - baseline_margin,
                        1]) # lower_left
        ]
        
        # transform warped corner points
        lower_left_border =  np.asarray([border[0], border[1], 1])
        upper_right_border = np.asarray([border[2], border[3], 1])
        lower_right_border = np.asarray([border[2], border[1], 1])
        upper_left_border =  np.asarray([border[0], border[3], 1])
        border_corners = [upper_left_border,upper_right_border,lower_right_border,lower_left_border]

        # calculate distance in px
        mae_px = calc_mae_px(query_corners,border_corners,transform)

        # calculate dist in metres
        truth_bbox = get_truth_bbox_dict(truth_bboxes, sheet_name)
        mae_m = calc_mae_m(query_corners,transform,image_path,truth_bbox)
        print(sheet_name,"MAE_px",mae_px,"mae m",mae_m)
        scores[sheet_name] = {
            "mae px":mae_px,
            "mae m":mae_m
            }
    return scores

def calc_and_dump(sheetsfile, out_dir):
    scores = calc_all_scores(sheetsfile, out_dir)
    scores_file = out_dir+"baseline_georef_scores.csv"
    dump_csv(scores_file, scores)

if __name__ == "__main__":
    out_dir = "E:/experiments/e2/"
    sheetsfile = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
    #sheetsfile = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed_wgs84.geojson"

    calc_and_dump(sheetsfile, out_dir)

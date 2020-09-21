import cv2
import argparse
from matplotlib import pyplot as plt

import find_sheet
import segmentation
import registration
from retrieval import retrieve_best_match

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path string")
    parser.add_argument("sheets", help="sheets json file path string", default="data/blattschnitt_dr100.geojson")
    parser.add_argument("--percent", help="colour balancethreshold", default=5, type=int)
    parser.add_argument("--noimg", help="set this flag to save resulting image files to disk", action="store_true")
    parser.add_argument("--plot", help="set this to true to show debugging plots", action="store_true")
    args = parser.parse_args()

    sheets_file = args.sheets
    bboxes = find_sheet.get_bboxes_from_json(sheets_file)
    # bboxes = bboxes[626:627] # gt-sheet 66 at idx 626
    bboxes = bboxes[124:125] # gt-sheet 259 at idx 125
    # bboxes = bboxes[90:93] # gt-sheet 258 at idx 92

    map_img = cv2.imread(args.input) # load map image

    water_mask = segmentation.extract_blue(map_img, args.percent) # extract rivers

    # find the bbox for this query image
    closest_image, closest_bbox, dist, score_list = retrieve_best_match(water_mask, bboxes)
    
    score_list= [(*s[:-1], find_sheet.find_name_for_bbox(sheets_file, bboxes[s[-1]])) for s in score_list]
    sheet_name = score_list[-1][-1] #find_sheet.find_name_for_bbox(sheets_file, closest_bbox)
    print("best sheet:", sheet_name, "with", dist)
    # print("ground truth at position:", len(score_list) - [s[-1] for s in score_list].index("66"))
    print("ground truth at position:", len(score_list) - [s[-1] for s in score_list].index("259"))
    # print("ground truth at position:", len(score_list) - [s[-1] for s in score_list].index("258"))


    if args.plot:
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        
        plt.subplot(2, 3, 1)
        map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
        plt.imshow(cv2.resize(map_img_rgb, (500,500)))
        plt.title("map image")
            
        plt.subplot(2, 3, 2)
        plt.gray()
        plt.imshow(cv2.resize(water_mask, (500,500)))
        plt.title("water mask from map")

        plt.subplot(2, 3, 3)
        plt.gray()
        plt.imshow(cv2.resize(closest_image, (500,500)))
        plt.title("closest reference rivers from OSM")

        plt.subplot(2, 1, 2)
        incidences = [s[0] for s in score_list]
        plt.hist(incidences, bins=max(incidences))
        plt.title("incidences of number of RANSAC inliers")
        
        # plt.subplot(3, 1, 3)
        # plt.plot(range(len(score_list)), [s[2] for s in score_list])
        # plt.title("template matching scores, sorted by #inliers")

        plt.show()

        # cv2.imshow("map_img", cv2.resize(map_img, (500,500)))
        # cv2.imshow("water mask from map", cv2.resize(water_mask, (500,500)))
        # cv2.imshow("closest reference rivers from OSM", cv2.resize(closest_image, (500,500)))

    if not args.noimg:
        cv2.imwrite("refimg_%s_%s.jpg" % (sheet_name, "-".join(map(str,closest_bbox))), closest_image)

        # align map image
        map_img_aligned = registration.align_map_image(map_img, water_mask, closest_image)
        
        if args.plot:
            plt.imshow(map_img_aligned)
            plt.title("aligned map")
            plt.show()
            # cv2.imshow("aligned map", map_img_aligned)

        # save aligned map image
        aligned_map_path = "aligned_%s_%s.jpg" % (sheet_name, "-".join(map(str,closest_bbox)))
        cv2.imwrite(aligned_map_path, map_img_aligned)

        # georeference aligned query image with bounding box
        registration.georeference(aligned_map_path, "georef_sheet_%s.tif" % sheet_name, closest_bbox)
        print("done!")
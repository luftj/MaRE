import cv2
import argparse

import find_sheet
import segmentation
import registration
from retrieval import retrieve_best_match

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path string")
    parser.add_argument("sheets", help="sheets json file path string", default="data/blattschnitt_dr100.geojson")
    parser.add_argument("percent", help="colour balancethreshold", default=5, type=int)
    parser.add_argument("--noimg", help="", action="store_true")
    args = parser.parse_args()

    sheets_file = args.sheets
    bboxes = find_sheet.get_bboxes_from_json(sheets_file)
    # bboxes = bboxes[620:630] # gt-sheet 66 at idx 626

    map_img = cv2.imread(args.input) # load map image

    water_mask = segmentation.extract_blue(map_img, args.percent) # extract rivers

    # find the bbox for this query image
    closest_image, closest_bbox, dist, score_list = retrieve_best_match(water_mask, bboxes)
    
    score_list= [(s[0],find_sheet.find_name_for_bbox(sheets_file, bboxes[s[1]])) for s in score_list]
    sheet_name = find_sheet.find_name_for_bbox(sheets_file, closest_bbox)
    print("best sheet:", sheet_name, "with", dist)
    print("ground truth at position:", len(score_list) - [s[1] for s in score_list].index("66"))

    # cv2.imshow("map_img", cv2.resize(map_img, (500,500)))
    # cv2.imshow("water mask from map", cv2.resize(water_mask, (500,500)))
    # cv2.imshow("closest reference rivers from OSM", cv2.resize(closest_image, (500,500)))

    cv2.imwrite("refimg_%s_%s.jpg" % (sheet_name, "-".join(map(str,closest_bbox))), closest_image)

    # align map image
    map_img_aligned = registration.align_map_image(map_img, water_mask, closest_image)
    
    cv2.imshow("aligned map", map_img_aligned)

    # save aligned map image
    aligned_map_path = "aligned_%s_%s.jpg" % (sheet_name, "-".join(map(str,closest_bbox)))
    cv2.imwrite(aligned_map_path, map_img_aligned)

    # georeference aligned query image with bounding box
    registration.georeference(aligned_map_path, "georef_sheet_%s.tif" % sheet_name, closest_bbox)

    cv2.waitKey(0)

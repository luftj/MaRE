
import sys
import cv2
import argparse
import logging
from datetime import datetime
from matplotlib import pyplot as plt
from numpy import fromfile, uint8

import find_sheet
import segmentation
import registration
from retrieval import retrieve_best_match

import config

def restrict__bboxes(sheets_path, target, num):
    # for debugging: don't check against all possible sheet locations, but only a subset
    bboxes = find_sheet.get_bboxes_from_json(sheets_path)
    idx = find_sheet.get_index_of_sheet(sheets_path, target)
    if idx is None:
        raise ValueError("ground truth name %s not found" % target)
    logging.debug("restricted bbox %s by %d around idx %s" %(target, num, idx))
    start = max(0, idx-(num//2))
    return bboxes[start:idx+(num//2)+1]

def process_sheet(img_path, sheets_path, cb_percent, plot=False, img=True, number=None, restrict=None, resize=None, rsize=None, crop=False):
    logging.info("Processing file %s with gt: %s" % (img_path,number))
    print("Processing file %s with gt: %s" % (img_path,number))

    if number and restrict:
        bboxes = restrict__bboxes(sheets_path, number, restrict)
    else:
        bboxes = find_sheet.get_bboxes_from_json(sheets_path)

    map_img = cv2.imdecode(fromfile(img_path, dtype=uint8), cv2.IMREAD_UNCHANGED)
    # map_img = cv2.imread(img_path) # load map image # WARNING: imread does not allow unicode file names!

    if resize:
        target_width = resize
        f = target_width / map_img.shape[1]
        target_height = int(f * map_img.shape[0])
        map_img = cv2.resize(map_img, (target_width, target_height), cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC) # area interpolation for downisizing, cubic upscaling

    water_mask = segmentation.extract_blue(map_img, cb_percent) # extract rivers

    # image size for intermediate processing
    processing_width = rsize if rsize else 500
    f = processing_width / map_img.shape[1]
    processing_height = int(f * map_img.shape[0])
    
    # find the best bbox for this query image
    closest_image, closest_bbox, dist, score_list = retrieve_best_match(water_mask, bboxes, (processing_width, processing_height))
    
    # find sheet name for prediction
    score_list = [(*s[:-1], find_sheet.find_name_for_bbox(sheets_file, bboxes[s[-1]])) for s in score_list]
    sheet_name = score_list[-1][-1] #find_sheet.find_name_for_bbox(sheets_file, closest_bbox)
    logging.info("best sheet: %s with score %f" %(sheet_name, dist))

    if number:
        logging.info("ground truth at position: %d" % (len(score_list) - [s[-1] for s in score_list].index(number)))

    if plot:
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

    if img:
        cv2.imwrite(config.path_output + "refimg_%s_%s.png" % (sheet_name, "-".join(map(str,closest_bbox))), closest_image)

        # align map image
        try:
            map_img_aligned, border = registration.align_map_image(map_img, water_mask, closest_image, (processing_width,processing_height), crop)
        except cv2.error as e:
            logging.warning("%s - could not register %s with prediction %s!" % (e, img_path, sheet_name))
            eval_entry = ["pred:"+sheet_name,"gt:"+number,"dist %d"%dist,"gt ar pos %d" % (len(score_list) - [s[-1] for s in score_list].index(number)),"registration: fail","correct: no"]
            logging.info("result: %s" % eval_entry)
            return 
        
        if args.plot:
            plt.imshow(map_img_aligned)
            plt.title("aligned map")
            plt.show()

        # save aligned map image
        aligned_map_path = config.path_output + "aligned_%s_%s.jpg" % (sheet_name, "-".join(map(str,closest_bbox)))
        logging.info("saved aligned image file to: %s" % aligned_map_path)
        cv2.imwrite(aligned_map_path, map_img_aligned, [cv2.IMWRITE_JPEG_QUALITY, config.jpg_compression])

        # georeference aligned query image with bounding box
        if crop:
            outpath = config.path_output + "georef_sheet_%s_crop.%s" % (sheet_name,config. output_file_ending)
            registration.georeference(aligned_map_path, outpath, closest_bbox)
        else:
            outpath = config.path_output + "georef_sheet_%s.%s" % (sheet_name,config. output_file_ending)
            registration.georeference(aligned_map_path, outpath, closest_bbox, border)
        logging.info("saved georeferenced file to: %s" % outpath)
    
    eval_entry = ["gt:"+number,"pred:"+sheet_name,"dist %f"%dist,"gt at pos %d"%(len(score_list) - [s[-1] for s in score_list].index(number)),"registration: success","correct %r"%(str(number)==str(sheet_name))]
    logging.info("result: %s" % eval_entry)

def process_list(list_path, sheets_path, cb_percent, plot=False, img=True, restrict=None, resize=None, rsize=None, crop=False):
    import os
    list_dir = os.path.dirname(list_path) + "/"
    with open(list_path, encoding="utf-8") as list_file:
        for line in list_file:
            line = line.strip()
            if not "," in line:
                logging.warning("skipping line: no ground truth given %s" % line)
                continue
            img_path, ground_truth = line.split(",")
            if not os.path.isabs(img_path[0]):
                img_path = os.path.join(list_dir,img_path)
            process_sheet(img_path, sheets_path, cb_percent, plot=plot, img=img, number=str(ground_truth), resize=resize, rsize=rsize, crop=crop, restrict=restrict)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path string or .txt list of images")
    parser.add_argument("sheets", help="sheets json file path string", default="data/blattschnitt_dr100.geojson")
    
    parser.add_argument("--gt", help="ground truth sheet name", default=None)
    parser.add_argument("--crop", help="set this to true to crop the map margins", action="store_true")
    
    parser.add_argument("--noimg", help="match sheet only, don't save registered image files", action="store_true")
    parser.add_argument("--plot", help="set this to true to show debugging plots", action="store_true")
    parser.add_argument("-v", help="set this to true to print log info to stdout", action="store_true")
    parser.add_argument("-ll", help="set this to get additional debug logging", action="store_true")

    parser.add_argument("--percent", help="colour balance threshold", default=5, type=int)
    parser.add_argument("--isize", help="resize input image to target width", default=None, type=int)
    parser.add_argument("--rsize", help="resize registration image to target width", default=None, type=int)    
    parser.add_argument("-r", help="restrict search space around ground truth", default=None, type=int)
    args = parser.parse_args()
    
    logging.basicConfig(filename=(config.path_logs + '/%s.log' % datetime.now().isoformat(timespec='minutes')).replace(":","-"), 
                        level=(logging.DEBUG if args.ll else logging.INFO), 
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-7.7s]  %(message)s") # gimme all your loggin'!
    if args.v:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("new experiment with: %s" % sys.argv)

    sheets_file = args.sheets
    
    if args.input[-4:] == ".txt":
        process_list(args.input, sheets_file, args.percent, plot=args.plot, img=(not args.noimg), resize=args.isize, rsize=args.rsize, crop=args.crop, restrict=args.r)
    else:
        process_sheet(args.input, sheets_file, args.percent, plot=args.plot, img=(not args.noimg), resize=args.isize, rsize=args.rsize, crop=args.crop, number=args.gt, restrict=args.r)

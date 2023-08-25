from json.decoder import JSONDecodeError
import joblib
from time import time
from matplotlib import pyplot as plt
import cv2
import numpy as np
import progressbar    
import os
import logging

import segmentation
import find_sheet, osm
import config

from annoy import AnnoyIndex

def convert_to_cv_keypoint(x, y, size=8.0, octave=1, response=1, angle=0.0):
    """Use this to convert to opencv syntax when using skimage feature detectors"""
    k = cv2.KeyPoint()
    k.pt=(y,x)
    k.response = response
    k.octave=octave
    k.size=size
    k.angle=angle
    return k

# initialise detectors
detector_dict = {
    "kaze_upright": cv2.KAZE_create(upright=True),
    "akaze_upright": cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT),
    # "surf_upright": cv2.SURF.create(upright=1),
    "sift": cv2.SIFT.create(),
    # "ski_fast": Skimage_fast_detector(min_dist=5,thresh=0),
    "cv_fast": cv2.FastFeatureDetector.create(),
    "orb": cv2.ORB_create()
    }

kp_detector = detector_dict[config.kp_detector]

if config.detector in ["ski_fast","cv_fast"]:
    raise ValueError("%s detector only detects keypoints, doesn't compute descriptors" % config.detector)

detector = detector_dict[config.detector]

def extract_features(image, first_n=None, plot=False):
    """Detect and extract features in given image.

    Arguments:
    image -- the image to extract features from,
    first_n -- the number of keypoints to use. Will be the keypoints with the highest response value. Set to None to use all keypoints. (default: None)

    Returns a list of keypoints and a list of descriptors """

    if detector == kp_detector:
        kps, dsc = detector.detectAndCompute(image, None)
    else:
        kps = kp_detector.detect(image)
        kps, dsc = detector.compute(image, kps)
    
    if len(kps) == 0:
        logging.error("no keypoints found!")
        return [],[]
        
    if first_n:
        kd = zip(kps,dsc)
        kd = sorted(kd, key=lambda x: x[0].response, reverse=True)[:first_n]
        kps, dsc = zip(*kd) # unzip

    if plot:
        vis_img1 = None
        vis_img1 = cv2.drawKeypoints(image,kps,vis_img1, 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                                #,color=(0,99,66))#(255,100,50))#(0,100,255))
        plt.imshow(vis_img1)
        plt.axis('off')
        plt.show()

    return kps, dsc

def resize_by_width(shape, new_width): # to do: move this to some central helper file. this shows up in a lot of places
    width = new_width
    f = width / shape[1]
    height = int(f * shape[0])
    return (width, height)

def restrict_bboxes(sheets_path, target, num):
    # for debugging: don't check against all possible sheet locations, but only a subset
    bboxes = find_sheet.get_bboxes_from_json(sheets_path)
    idx = find_sheet.get_index_of_sheet(sheets_path, target)
    if idx is None:
        raise ValueError("ground truth name %s not found" % target)
    print("restricted bbox %s by %d around idx %s" %(target, num, idx))
    start = max(0, idx-(num//2))
    return bboxes[start:idx+(num//2)+1]

sheet_names = {}

def build_index(sheets_path, restrict_class=None, restrict_range=None, store_desckp=True):
    print("building index...")

    if restrict_class and restrict_range:
        bboxes = restrict_bboxes(sheets_path, restrict_class, restrict_range)
    else:
        bboxes_dict = find_sheet.get_dict(sheets_path)

    bboxes = list(bboxes_dict.values())

    keypoint_dict = {}
    t = AnnoyIndex(config.index_descriptor_length, config.index_annoydist)
    t.on_disk_build(config.reference_index_path)
    idx_id = 0
    
    print("populating index...")
    index_dict = {}
    progress = progressbar.ProgressBar(maxval=len(bboxes))
    for bbox in progress(bboxes):
        try:
            rivers_json = osm.get_from_osm(bbox)
        except JSONDecodeError:
            print("error in OSM data for bbox %s, skipping sheet" % bbox)
            continue
        reference_river_image = osm.paint_features(rivers_json, bbox)

        # reduce image size for performance with fixed aspect ratio
        processing_size = resize_by_width(reference_river_image.shape, config.index_img_width_train)
        reference_image_small = cv2.resize(reference_river_image, processing_size, config.resizing_index_building)
        if config.index_border_train:
            reference_image_small = cv2.copyMakeBorder(reference_image_small, 
                                        config.index_border_train, config.index_border_train, config.index_border_train, config.index_border_train, 
                                        cv2.BORDER_CONSTANT, None, 0)
        # get class label
        class_label = list(bboxes_dict.keys())[bboxes.index(bbox)]
        if not class_label:
            print("error in class name. skipping bbox", bbox)
            continue

        # extract features of sheet
        try:
            keypoints, descriptors = extract_features(reference_image_small, first_n=config.index_n_descriptors_train)
        except ValueError as e:
            print(type(e),e)
            print("error in descriptors. skipping sheet", class_label)
            continue
        except cv2.error as e:
            print(e)
            print(class_label,bbox)
            exit()
        if descriptors is None or len(descriptors)==0 or descriptors[0] is None:
            print("no descriptors in bbox ",bbox)
            print("error in descriptors. skipping sheet", class_label)
            continue
        # add features and class=sheet to index
        index_dict[class_label] = descriptors
        keypoint_dict[class_label] = [x.pt for x in keypoints]

        for x in descriptors:
            t.add_item(idx_id, x)
            idx_id += 1
        sheet_names[class_label] = len(descriptors)

    print("compiling tree...")
    t.build(config.index_num_trees, n_jobs=-1) # compile index and save to disk
    # save other data to disk to disk
    print("saving to disk...")
    joblib.dump(sheet_names, config.reference_sheets_path)
    if store_desckp:
        for sheet, descs in index_dict.items():
            joblib.dump(descs, config.reference_descriptors_folder+"/%s.clf" % sheet)
        for sheet, kps in keypoint_dict.items():
            joblib.dump(kps, config.reference_keypoints_folder+"/%s.clf" % sheet)

def predict_annoy(descriptors):
    u = AnnoyIndex(config.index_descriptor_length, config.index_annoydist)
    u.load(config.reference_index_path) # super fast, will just mmap the file
    from annoy_helper import get_sheet_for_id, sheets

    votes = {k:0 for k in sheets}
    for desc in descriptors:
        # will find the k nearest neighbors
        NN_ids = u.get_nns_by_vector(desc, config.index_k_nearest_neighbours, include_distances=True) # will find the n nearest neighbors
        distances = NN_ids[1]
        NN_ids = NN_ids[0]

        if config.index_lowes_test_ratio:
            if min(distances) < config.index_lowes_test_ratio * max(distances):
                # good match
                NN_ids = [NN_ids[0]]
            else:
                continue

        NN_names = [get_sheet_for_id(i) for i in NN_ids]
        # vote for the nearest neighbours (codebook response)
        for name in NN_names:
            if config.index_voting_scheme == "antiprop":
                votes[name] += 1/(NN_names.index(name)+1) # antiproportional weighting
            else:
                # todo: allow other voting schemes in config
                raise NotImplementedError("voting scheme '%s' not implemented" % config.index_voting_scheme)
                
    if votes == {}:
        print("truth not in index")
        return -1

    votes = sorted(votes.items(),key=lambda x:x[1], reverse=True)
    # print("truth:",class_label_truth,"index:",[x[0] for x in votes].index(class_label_truth))

    return votes # most similar prediction is in 0th position

def search_in_index(img_path, class_label_truth, imgsize=None):
    """
    Predict the identity of a single map located at img_path.
    """
    # usgs name fixes:
    class_label_truth = class_label_truth.replace("Mts","Mountains")
    class_label_truth = class_label_truth.replace("Mtns","Mountains")
    # class_label_truth = class_label_truth.replace(" Of "," of ")
    class_label_truth = class_label_truth.replace("St ","Saint ")
    class_label_truth = class_label_truth.replace(" Du "," du ")

    # load query sheet
    try:
        map_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if imgsize:
            scale = imgsize / map_img.shape[0] # keep aspect by using width factor
            map_img = cv2.resize(map_img, None, 
                                fx=scale, fy=scale,
                                interpolation=config.resizing_index_query)
    except FileNotFoundError:
        print("couldn't find input file at %s" % img_path)
        return -1
    if len(map_img.shape)>2:
        print("segmenting")
        # seegment query sheet
        water_mask = segmentation.extract_blue(map_img) # extract rivers
    else:
        # grayscale image - already segmented
        water_mask = map_img
    # image size for intermediate processing
    processing_size = resize_by_width(map_img.shape, config.index_img_width_query)
    water_mask_small = cv2.resize(water_mask, processing_size, interpolation=config.resizing_index_query)
    # extract features from query sheet
    kps, descriptors = extract_features(water_mask_small, first_n=config.index_n_descriptors_query)
    # set up features as test set    
    prediction = predict_annoy(descriptors)

    try:
        gt_index = [str(x[0]) for x in prediction].index(class_label_truth)
        print("Prediction", prediction[0][0], "Truth at index", gt_index)
        # print("Prediction score %.3f Truth score %.3f" % (prediction[0][1], prediction[gt_index][1]))
        return gt_index
    except Exception as e:
        print("truth not in index")
        return -1

def search_list(list_path):
    # iterate over all sheets in list
    positions = []
    labels = []

    with open(list_path, encoding="utf-8") as list_file:
        for line in list_file:
            line = line.strip()
            print(line)
            if not "," in line:
                print("skipping line: no ground truth given %s" % line)
                continue
            img_path, class_label = line.rsplit(",",1)
            if not os.path.isabs(img_path[0]):
                list_dir = os.path.dirname(list_path) + "/"
                img_path = os.path.join(list_dir,img_path)
            pos = search_in_index(img_path, class_label)
            positions.append(pos)
            labels.append(class_label)
    
    return zip(labels,positions)

def profile_index_building(sheets_path_reference, outfolder):
    # profile index building
    import cProfile, pstats
    from pstats import SortKey
    prf_file = outfolder + "/profile_indexbuild.prf"
    # change paths in config
    config.reference_sheets_path = outfolder + "/sheets.clf"
    config.reference_index_path  = outfolder + "/index.ann"
    cProfile.run('build_index("%s", store_desckp=False)' % sheets_path_reference, prf_file)

    p = pstats.Stats(prf_file)
    # p.print_stats()
    # p.sort_stats(-1).print_stats()
    # p.sort_stats(SortKey.NAME)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(30)
    # p.sort_stats(SortKey.TIME).print_stats(30)
    # p.sort_stats(SortKey.TIME).print_stats(function_of_interest)
    # p.print_callers(1,"sort")

def reproject_all_osm():
    """in case you have a really weird projection in your query data, it might make sense to project reference maps to the SRS of the query maps (so the images stay rectangular)"""
    osm_path = config.path_osm
    outpath = config.path_osm[:-1]+"_reproj/"
    os.makedirs(outpath, exist_ok=True)
    t0 = time()
    for osm_file in os.listdir(osm_path):
        in_file = osm_path + osm_file
        out_file = outpath + osm_file
        command = 'ogr2ogr -f "GeoJSON" %s %s -s_srs "%s" -t_srs "%s"' % (out_file, in_file, config.proj_osm, config.proj_map)
        os.system(command)
    t1 = time()
    print("time for convert: %f" % (t1-t0))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sheets", help="sheets json file path string", default="data/blattschnitt_dr100.geojson")
    parser.add_argument("--output", help="path to output profiling files", default=None)
    parser.add_argument("--list", help="path to image list to test indexing", default=None)
    parser.add_argument("--rebuild", help="set this to true to rebuild the index", action="store_true")
    args = parser.parse_args()
    
    # reproject_all_osm()
    # exit()

    # make logdir
    os.makedirs(config.path_logs, exist_ok=True)
    logging.basicConfig(filename=config.path_logs+'/indexing.log', level=logging.INFO) # gimme all your loggin'!

    if args.output:
        print("profiling index building...")
        os.makedirs(args.output, exist_ok=True)
        profile_index_building(args.sheets, args.output)
        exit()

    if args.rebuild:
        print("rebuilding index...")
        # create folders first
        os.makedirs(config.reference_descriptors_folder, exist_ok=True)
        os.makedirs(config.reference_keypoints_folder, exist_ok=True)

        build_index(args.sheets)

    if args.list:
        print("evaluating index...")
        lps = search_list(args.list)

        positions = []
        with open("index_result.csv","w") as fp:
            for l,p in lps:
                fp.write("%s : %d\n"%(l,p))
                positions.append(p)

        print("mean index position:", sum(positions)/len(positions))
        print("median index position:", sorted(positions)[len(positions)//2])

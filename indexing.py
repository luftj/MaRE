import joblib
from time import time
import random
from matplotlib import pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
import cv2
import numpy as np
import progressbar    
import os
import logging

import segmentation
import find_sheet, osm

logging.basicConfig(filename='dump.log', level=logging.INFO) # gimme all your loggin'!

plot = False

lowes_test_ratio = None#0.8
n_matches = 100
score_eq = "avg_score" # "num_match"
norm = cv2.NORM_L2
cross_check = False
n_descriptors_query = 300
img_width_query = 500

if lowes_test_ratio and cross_check:
    raise ValueError("can't put cross-check with lowe's test")

# the following parameters require rebuilding the index
rebuild_index = False
n_descriptors_train = 300
img_width_train = 500
# kp_detector = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT)
# detector = cv2.xfeatures2d_LATCH.create(rotationInvariance=False, half_ssd_size=3)
# detector = cv2.xfeatures2d_LATCH.create(rotationInvariance=False)
detector = cv2.xfeatures2d_SURF.create(upright=1)
kp_detector = cv2.xfeatures2d_SURF.create(upright=1)
# kp_detector = cv2.KAZE_create(upright=True)
# kp_detector = cv2.FastFeatureDetector.create()

def patch_features(image, scale_range=None, binary=False):
    # detector = cv2.FastFeatureDetector.create()
    detector = cv2.KAZE.create(upright=True)

    # find and draw the keypoints
    keypoints = detector.detect(image)
    # print([k.size for k in keypoints])
    # print([k.octave for k in keypoints])
    # print([k.response for k in keypoints])
    # pad = scale_range[-1] if scale_range else 30
    pad = scale_range[-1] if scale_range else 30#max([int(2*k.size) for k in keypoints])
    # exit()
    # image2 = cv2.drawKeypoints(image, keypoints, None, color=(255,0,0))
    # cv2.imshow("ksp",image2)
    # cv2.waitKey(-1)
    first_n=300
    keypoints = sorted(keypoints, key=lambda x: -x.response)[:first_n]

    image_border = cv2.copyMakeBorder(image, 
                                    pad, pad, pad, pad, 
                                    cv2.BORDER_CONSTANT, None, 0)
    
    patch_size = 8
    descriptors = []
    for corner in keypoints:
        x,y = corner.pt
        x += pad
        y += pad # adjust for padding padding
        
        if scale_range:
            patch_size_now = random.uniform(*scale_range)
        else:
            patch_size_now = int(2*corner.size) # patch_size

        patch = image_border[int(y-patch_size_now):int(y+patch_size_now),int(x-patch_size_now):int(x+patch_size_now)]

        if True:#scale_range:
            #resize to 16x16
            # print(patch.shape)
            # print(2*patch_size,2*patch_size)
            patch = cv2.resize(patch,(2*patch_size,2*patch_size))

        descriptor = patch.flatten()
        if cv2.countNonZero(descriptor) < 24:
            continue
        if binary:
            descriptor = descriptor > 0
            # descriptor.astype(np.int)
        descriptors.append(descriptor)
        if descriptor.shape[0] != 256:
            print("irregular patch",descriptor.shape[0],x,y,patch_size_now,image_border.shape)
        # print(descriptor, descriptor.shape)

    # print(len(keypoints), descriptors[0])
    # exit()
    
    # n_descs = 500
    # if len(descriptors) > n_descs:
    #     descriptors = random.sample(descriptors, n_descs)
    # exit()
    descriptors = np.vstack(descriptors)
    return keypoints, descriptors

def extract_features(image, first_n=None):
    # return patch_features(image)
    # first_n = None
    if first_n:
        kps = kp_detector.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:first_n]
        kps, dsc = detector.compute(image, kps)  # todo: use cv2.detectAndCompute instead, is faster
    else:
        kps, dsc = detector.detectAndCompute(image, None)
    # print(dsc)
    # exit()
    return kps, dsc

def resize_by_width(shape, new_width):
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

def build_index(rsize=None, restrict_class=None, restrict_range=None):
    print("building index...")
    t0 = time()

    sheets_path = "data/blattschnitt_dr100_regular.geojson"
    if restrict_class and restrict_range:
        bboxes = restrict_bboxes(sheets_path, restrict_class, restrict_range)
    else:
        bboxes = find_sheet.get_bboxes_from_json(sheets_path)

    X = []
    Y = []

    index_dict = {}
    keypoint_dict = {}
    
    progress = progressbar.ProgressBar(maxval=len(bboxes))
    for bbox in progress(bboxes):
        rivers_json = osm.get_from_osm(bbox)
        reference_river_image = osm.paint_features(rivers_json, bbox)

        # reduce image size for performance with fixed aspect ratio
        processing_size = resize_by_width(reference_river_image.shape, rsize if rsize else 500)
        reference_image_small = cv2.resize(reference_river_image, processing_size, cv2.INTER_AREA)
        # get class label
        class_label = find_sheet.find_name_for_bbox(sheets_path, bbox)
        if not class_label:
            print("error in class name. skipping bbox", bbox)
            continue

        # extract features of sheet
        try:
            keypoints, descriptors = extract_features(reference_image_small, first_n=n_descriptors_train)
        except Exception as e:
            print(e)
            print("error in descriptors. skipping sheet", class_label)
            # exit()
            continue
        if descriptors is None or len(descriptors)==0 or descriptors[0] is None:
            print("error in descriptors. skipping sheet", class_label)
            continue
        # add features and class=sheet to index
        # X.extend(descriptors)
        # Y.extend([class_label]*len(descriptors))
        index_dict[class_label] = descriptors
        keypoint_dict[class_label] = [x.pt for x in keypoints]
    # clf = RandomForestClassifier(n_estimators=100, verbose=True, n_jobs=-1)
    # clf = clf.fit(X, Y)
    
    t1 = time()
    print("building index took %f seconds. %f s per sheet" % (t1-t0,(t1-t0)/len(bboxes)))

    # save index to disk
    joblib.dump(index_dict, "index.clf", compress=3)
    joblib.dump(keypoint_dict, "keypoints.clf", compress=3)
    print("compress and store time: %f s" % (time()-t1))
    # return clf
    return index_dict

bf = None

def predict(sample, clf):
    # prediction_class = clf.predict(sample)
    # prediction = clf.predict_proba(sample)
    prediction = []
    match_dict = {}
    progress = progressbar.ProgressBar(maxval=len(clf.keys()))
    
    # create BFMatcher object
    bf = cv2.BFMatcher(norm, crossCheck=cross_check)
    # FLANN_INDEX_KDTREE = 1
    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm =
    #             # FLANN_INDEX_LSH,
    #             # table_number = 6, # 12
    #             # key_size = 12,     # 20
    #             # multi_probe_level = 1) #2
    #             FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)   # or pass empty dictionary
    # global bf
    # if not bf:
    #     bf = cv2.FlannBasedMatcher(index_params,search_params)
    #     print("add")
    #     bf.add([ x for x in clf.values() if len(x)>0])
    #     print("train")
    #     bf.train()

    for label in progress(clf.keys()):
        # if label != "258":
        #     continue

        # Match descriptors.
        if not lowes_test_ratio:
            matches = bf.match(sample, clf[label]) # simple matching
        else:
            # Lowe's test ratio refinement
            nn_matches = bf.knnMatch(sample, clf[label], 2)
            if len(nn_matches[0]) < 2:
                continue
            matches = []
            nn_match_ratio = lowes_test_ratio # Nearest neighbor matching ratio
            for m,n in nn_matches:
                if m.distance < nn_match_ratio * n.distance:
                    matches.append(m)
        
        # print("#matches", len(matches))
        if n_matches:
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
            matches = matches[:n_matches]

        if plot:
            idx = 1
            for m in (matches[:10]):
                plt.subplot(10,2,idx)
                plt.imshow(np.reshape(r[m.trainIdx],(16,16)))
                plt.title(int(m.distance))
                idx+=1
                plt.subplot(10,2,idx)
                plt.imshow(np.reshape(q[m.queryIdx],(16,16)))
                idx+=1
            plt.show()

        ### reverse matching
        # matches = bf.match(clf[label], sample)
        # nn_matches_reverse = bf.knnMatch(clf[label], sample, 2)
        # for m,n in nn_matches_reverse:
        #     if m.distance < nn_match_ratio * n.distance:
        #         matches.append(m)

        # matches2 = bf.match(clf[label], sample)
        # matches2 = sorted(matches2, key = lambda x:x.distance)
        # matches2 = matches2[:100]
        # matches.extend(matches2)

        match_dict[label] = matches
        distances = [x.distance for x in matches]

        if len(matches) == 0:
            prediction.append((label,1))
        else:
            if score_eq == "num_match":
                score = len(matches)
            elif score_eq == "avg_score":
                score = sum(distances)/len(matches)
            prediction.append((label,score))

    prediction.sort(key=lambda x: x[1], reverse=(score_eq=="num_match"))
    prediction_class = prediction[0][0]
    return prediction_class, prediction, match_dict

def search_in_index(img_path, class_label_truth, cb_percent=5, clf=None):
    # load query sheet
    map_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # seegment query sheet
    water_mask = segmentation.extract_blue(map_img, cb_percent) # extract rivers
    # image size for intermediate processing
    rsize = img_width_query
    processing_size = resize_by_width(map_img.shape, rsize if rsize else 500)
    water_mask_small = cv2.resize(water_mask, processing_size, interpolation=cv2.INTER_AREA)
    # extract features from query sheet
    kps, descriptors = extract_features(water_mask_small, first_n=n_descriptors_query)
    # set up features as test set
    # load index from disk
    if not clf:
        clf = joblib.load("index.clf")  
    # classify sheet with index
    prediction_class, prediction, _ = predict(descriptors, clf)
    
    # probabilities = list(zip(clf.classes_,prediction[0]))

    # probabilities.sort(key=lambda x: x[1], reverse=True)
    # print(probabilities)
    try:
        gt_index = [x[0] for x in prediction].index(class_label_truth)
        print("Prediction", prediction_class, "Truth at index", gt_index)
        return gt_index
    except:
        print("truth not in index")
        return -1

def search_list(list_path, clf=None):
    # iterate over all sheets in list
    t0 = time()
    positions = []
    with open(list_path, encoding="utf-8") as list_file:
        for line in list_file:
            line = line.strip()
            print(line)
            if not "," in line:
                print("skipping line: no ground truth given %s" % line)
                continue
            img_path, class_label = line.split(",")
            if not os.path.isabs(img_path[0]):
                list_dir = os.path.dirname(list_path) + "/"
                img_path = os.path.join(list_dir,img_path)
            pos = search_in_index(img_path, class_label, clf=clf)
            positions.append(pos)
            
    t1 = time()
    print("searching list took %f seconds. %f s per sheet" % (t1-t0,(t1-t0)/len(positions)))
    return positions

if __name__ == "__main__":
    if rebuild_index:
        clf = build_index(rsize=img_width_train)#restrict_class="524", restrict_range=200)

    # img_path = "E:/data/deutsches_reich/SBB/cut/SBB_IIIC_Kart_L 1330_Blatt 669 von 1908.tif"#SBB_IIIC_Kart_L 1330_Blatt 259 von 1925_koloriert.tif"
    # class_label_truth = "669"
    # search_in_index(img_path, class_label_truth)

    list_path = "E:/data/deutsches_reich/SBB/cut/list.txt"
    positions = search_list(list_path)#, clf=clf)
    print(positions)
    # positions200 = [352, 414, 2, 6, 0, 2, 3, 0, 2, 155, 229, 104, 326, 20, 2, 4,  69, 36, 0, 5, 2, 2, 3, 33, 28, 0, 8, 2, 326, 2, 3, 2, 3]
    # positions300 = [399, 311, 1, 2, 0, 3, 5, 0, 4,  68, 180, 117, 467,  3, 2, 4, 125, 28, 0, 6, 2, 2, 6, 38, 33, 0, 3, 2, 227, 2, 3, 2, 3]
    # positions400 = [432, 192, 1, 3, 0, 3, 5, 2, 4,  81, 166, 185, 435,  4, 2, 4, 125, 84, 0, 4, 1, 3, 26, 60, 33, 0, 3, 12, 230, 2, 3, 2, 3]
    maha = [3.92,4.17,63.62,18.6,37.51,35.58,15.36,14.71,24.17,15.68,11.55,7.6,4.52,10.98,24.36,25.53,6.96,11.7,28.92,36.52,37.75,21.36,25.05,4.76,11.95,8.6,15.76,11.58,2.7,42.41,40.9,53.08,13.34]
    plt.boxplot([x for x in positions if x >= 0])
    # plt.scatter(maha, positions300, c="r")
    # plt.scatter(maha, positions200, c="b")
    # plt.scatter(maha, positions400, c="g")
    # plt.show()

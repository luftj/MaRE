import joblib
from time import time
import random
from matplotlib import pyplot as plt
import cv2
import numpy as np
import progressbar    
import os
import logging
# from extract_patches.core import extract_patches

import segmentation
import find_sheet, osm

from annoy import AnnoyIndex

class Skimage_fast_detector:
    def __init__(self, min_dist, thresh):
        self.min_dist = min_dist
        self.thresh = thresh

    def detect(self, image):
        from skimage.feature import corner_fast, corner_peaks

        keypoints = corner_peaks(corner_fast(image), min_distance=self.min_dist, threshold_rel=self.thresh)
        random.shuffle(keypoints) # in case we want to limit the number of keypoints. since they don't have a response to sort by
        keypoints = [convert_to_cv_keypoint(x,y) for (x,y) in keypoints]
        return keypoints
    
    def detectAndCompute(image, kps):
        raise NotImplementedError("this detector only detects, doesn't compute")
    def compute(image, kps):
        raise NotImplementedError("this detector only detects, doesn't compute")

class Patch_extractor:
    def __init__(self, patch_size_desc, patch_mag, minArea=0.15, maxArea=0.9, binary=False, plot=False):
        self.binary = binary
        self.plot = plot
        self.patch_size_desc = patch_size_desc
        self.minArea = minArea # in percent
        self.maxArea = maxArea
        self.patch_mag = patch_mag

    def detect(self, image):
        raise NotImplementedError("this detector only computes, doesn't detect")
    
    def detectAndCompute(image, kps):
        raise NotImplementedError("this detector only detects, doesn't compute")
    def compute(self, image, keypoints):
        """Extract patch features around given keypoints from an image"""
        patches = extract_patches(keypoints, image, self.patch_size_desc, self.patch_mag)

        patches = [p for p in patches if (self.patch_size_desc**2)*self.maxArea > np.count_nonzero(p) > (self.patch_size_desc**2)*self.minArea]

        if self.binary:
            patches = [(p>0).astype(np.uint8) for p in patches]

        if self.plot:
            vis_img1 = None
            vis_img1 = cv2.drawKeypoints(image,keypoints,vis_img1, 
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.imshow(vis_img1)
            plt.show()
            show_idx = 0
            fig = plt.figure()
            num_plots_root = 5
            for i in range(1,num_plots_root**2+1):
                fig.add_subplot(num_plots_root, num_plots_root, i) 
                plt.imshow(patches[show_idx+i])
                plt.xticks([])
                plt.yticks([])
                plt.title("r:%f s:%f" % (keypoints[i].response, keypoints[i].size))
            plt.show()

        descriptors = [p.flatten() for p in patches]
        descriptors = np.vstack(descriptors)
        return keypoints, descriptors

class Patch_extractor_2:
    def __init__(self, patch_size_desc, scale_range, minArea=0.15, maxArea=0.9, binary=False, plot=False):
        self.binary = binary
        self.plot = plot
        self.patch_size_desc = patch_size_desc
        self.minArea = minArea # in percent
        self.maxArea = maxArea
        self.scale_range = scale_range

    def detect(self, image):
        raise NotImplementedError("this detector only computes, doesn't detect")
    def detectAndCompute(image, kps):
        raise NotImplementedError("this detector only detects, doesn't compute")

    def compute(self, image, keypoints):
        """Extract patch features around given keypoints from an image"""
        # pad = scale_range[-1] if scale_range else 30
        pad = self.scale_range[-1] if self.scale_range else 30#max([int(2*k.size) for k in keypoints])
        # image2 = cv2.drawKeypoints(image, keypoints, None, color=(255,0,0))

        image_border = cv2.copyMakeBorder(image, 
                                        pad, pad, pad, pad, 
                                        cv2.BORDER_CONSTANT, None, 0)
        
        patch_size = self.patch_size_desc // 2
        descriptors = []
        for corner in keypoints:
            x,y = corner.pt
            x += pad
            y += pad # adjust for padding padding
            
            if self.scale_range:
                patch_size_now = random.uniform(*self.scale_range)
            else:
                patch_size_now = patch_size#int(2*corner.size) # patch_size

            patch = image_border[int(y-patch_size_now):int(y+patch_size_now),int(x-patch_size_now):int(x+patch_size_now)]

            if self.scale_range:
                #resize to 16x16
                # print(patch.shape)
                # print(2*patch_size,2*patch_size)
                patch = cv2.resize(patch,(2*patch_size,2*patch_size), interpolation=cv2.INTER_AREA)

            descriptor = patch.flatten()
            if np.count_nonzero(descriptor) < ((patch_size*2)**2)*self.minArea:
                continue
            if self.binary:
                descriptor = descriptor > 0
                descriptor.astype(np.uint8)
            descriptors.append(descriptor)
            if descriptor.shape[0] != (patch_size*2)**2:
                print("irregular patch",descriptor.shape[0],x,y,patch_size_now,image_border.shape)

        if self.plot:
            vis_img1 = None
            vis_img1 = cv2.drawKeypoints(image,keypoints,vis_img1, 
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.imshow(vis_img1)
            plt.show()
            show_idx = 0
            fig = plt.figure()
            num_plots_root = 5
            patches = [np.reshape(d,(self.patch_size_desc,self.patch_size_desc)) for d in descriptors]
            for i in range(1,num_plots_root**2+1):
                fig.add_subplot(num_plots_root, num_plots_root, i) 
                plt.imshow(patches[show_idx+i])
                plt.xticks([])
                plt.yticks([])
                plt.title("r:%f s:%f" % (keypoints[i].response, keypoints[i].size))
            plt.show()


        descriptors = np.vstack(descriptors)
        return keypoints, descriptors

def convert_to_cv_keypoint(x, y, size=8.0, octave=1, response=1, angle=0.0):
    k = cv2.KeyPoint()
    k.pt=(y,x)
    k.response = response
    k.octave=octave
    k.size=size
    k.angle=angle
    return k


logging.basicConfig(filename='logs/dump.log', level=logging.INFO) # gimme all your loggin'!

plot = False

lowes_test_ratio = None#0.8
n_matches = 100
score_eq = "avg_score" # "num_match"
norm = cv2.NORM_L2
cross_check = False
n_descriptors_query = 500
img_width_query = 500

if lowes_test_ratio and cross_check:
    raise ValueError("can't do cross-check with lowe's test")

# the following parameters require rebuilding the index
rebuild_index = True
annoydist = "euclidean"
n_descriptors_train = 300
img_width_train = 500
border_train = 30
# detector = cv2.xfeatures2d_LATCH.create(rotationInvariance=False, half_ssd_size=3)
# detector = cv2.xfeatures2d_LATCH.create(rotationInvariance=False)
patch_size = 30 # relevant for plotting
# detector = Patch_extractor(patch_size_desc=patch_size, minArea=0.15, maxArea=0.9, 
#                             plot=False, patch_mag = 8.0)
# detector = Patch_extractor_2(patch_size_desc=patch_size, scale_range=[30,30], minArea=0.1, maxArea=0.8)
# detector = cv2.xfeatures2d_SURF.create(upright=1)
# detector = kp_detector = cv2.xfeatures2d_SURF.create(upright=1)
# detector = kp_detector = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT)
detector = kp_detector = cv2.KAZE_create(upright=True)
# kp_detector = cv2.AgastFeatureDetector.create()
# kp_detector = cv2.FastFeatureDetector.create()
# kp_detector = Skimage_fast_detector(min_dist=5,thresh=0)

def extract_features(image, first_n=None):
    """Detect and extract features in given image.

    Arguments:
    image -- the image to extract features from,
    first_n -- the number of keypoints to use. Will be the keypoints with the highest response value. Set to None to use all keypoints. (default: None)

    Returns a list of keypoints and a list of descriptors """

    if first_n:
        kps = kp_detector.detect(image)
        kps = sorted(kps, key=lambda x: x.response, reverse=True)[:first_n]
        
        kps, dsc = detector.compute(image, kps)  # prefer cv2.detectAndCompute instead, is faster
    else:
        if detector == kp_detector:
            kps, dsc = detector.detectAndCompute(image, None)
        else:
            kps = kp_detector.detect(image)
            kps, dsc = detector.compute(image, kps)

    if plot:
        vis_img1 = None
        vis_img1 = cv2.drawKeypoints(image,kps,vis_img1, 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(vis_img1)
        plt.show()

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

sheet_names = {}

from dask import delayed

def build_index(rsize=None, restrict_class=None, restrict_range=None):
    print("building index...")
    t0 = time()

    sheets_path = "data/blattschnitt_dr100_regular.geojson"
    if restrict_class and restrict_range:
        bboxes = restrict_bboxes(sheets_path, restrict_class, restrict_range)
    else:
        bboxes = find_sheet.get_bboxes_from_json(sheets_path)

    keypoint_dict = {}

    t = AnnoyIndex(64, annoydist)
    t.on_disk_build("index.ann")
    idx_id = 0
    
    # def calc(bboxes):
    index_dict = {}
    progress = progressbar.ProgressBar(maxval=len(bboxes))
    for bbox in progress(bboxes):
    # for bbox in bboxes:
        rivers_json = osm.get_from_osm(bbox)
        reference_river_image = osm.paint_features(rivers_json, bbox)

        # reduce image size for performance with fixed aspect ratio
        processing_size = resize_by_width(reference_river_image.shape, rsize if rsize else 500)
        reference_image_small = cv2.resize(reference_river_image, processing_size, cv2.INTER_AREA)
        if border_train:
            reference_image_small = cv2.copyMakeBorder(reference_image_small, 
                                        border_train, border_train, border_train, border_train, 
                                        cv2.BORDER_CONSTANT, None, 0)
        # get class label
        class_label = find_sheet.find_name_for_bbox(sheets_path, bbox)
        if not class_label:
            print("error in class name. skipping bbox", bbox)
            continue

        # extract features of sheet
        try:
            keypoints, descriptors = extract_features(reference_image_small, first_n=n_descriptors_train)
        except ValueError as e:
            print(type(e),e)
            print("error in descriptors. skipping sheet", class_label)
            continue
        if descriptors is None or len(descriptors)==0 or descriptors[0] is None:
            print("no descriptors in bbox ",bbox)
            print("error in descriptors. skipping sheet", class_label)
            continue
        # add features and class=sheet to index
        index_dict[class_label] = descriptors
        # keypoint_dict[class_label] = [x.pt for x in keypoints]

        for x in descriptors:
            t.add_item(idx_id, x)
            idx_id += 1
        sheet_names[class_label] = len(descriptors)
    
    # index_dictd = delayed(calc)(bboxes)
    # from dask.diagnostics import ProgressBar
    # ProgressBar().register()
    # with ProgressBar():
    #     index_dict = index_dictd.compute()
    # for class_label, descriptors in index_dict.items():
    #     sheet_names[class_label] = len(descriptors)
    #     for x in descriptors:
    #         t.add_item(idx_id, x)
    #         idx_id += 1

    t1 = time()
    print("building index took %f seconds. %f s per sheet" % (t1-t0,(t1-t0)/len(bboxes)))

    t.build(10, n_jobs=-1)
    # t.save("index.ann")
    # save index to disk
    joblib.dump(sheet_names, "sheets.clf")
    # joblib.dump(index_dict, "index.clf", compress=3)
    # joblib.dump(keypoint_dict, "keypoints.clf", compress=3)
    print("compress and store time: %f s" % (time()-t1))
    # return clf
    return index_dict

bf = None

def predict(sample, clf, truth=None):
    """Predict the similarities of the descriptors in the index.
    
    Arguments:
    sample -- the descriptors of the query image.
    clf -- the index (a dict with 'image name':[descriptors])

    returns a tuple with 
    - the predicted image name, 
    - a list of all possible images ordered by similarity, 
    - a dict with the matches for each image in the index.
    """
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
    # trainTime = time()
    # global bf
    # if not bf:
    #     bf = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)
    #     # bf = cv2.FlannBasedMatcher(index_params,search_params)
    #     print("add")
    #     bf.add([ x for x in clf.values() if len(x)>0])
    #     print("train")
    #     bf.train()
    # print("FLANN training time %0.2fs" % (time()-trainTime))

    for label in progress(clf.keys()):
        # if label != truth:
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
                plt.imshow(np.reshape(clf[label][m.trainIdx],(patch_size,patch_size)))
                plt.title(int(m.distance))
                idx+=1
                plt.subplot(10,2,idx)
                plt.imshow(np.reshape(sample[m.queryIdx],(patch_size,patch_size)))
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
    u = AnnoyIndex(64, annoydist)
    u.load('index.ann') # super fast, will just mmap the file
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
        clf = joblib.load("sheets.clf")  


    
    from annoytest import get_sheet_for_id
    votes = {k:0 for k in clf.keys()}
    for desc in descriptors:
        # NN_ids = u.get_nns_by_vector(desc, 2) # will find the n nearest neighbors
        NN_ids = u.get_nns_by_vector(desc, 50, include_distances=True) # will find the n nearest neighbors
        distances = NN_ids[1]
        NN_ids = NN_ids[0]

        if lowes_test_ratio:
            if min(distances) < lowes_test_ratio * max(distances):
                # good match
                NN_ids = [NN_ids[0]]
            else:
                continue

        NN_names = [get_sheet_for_id(clf,i) for i in NN_ids]
        # print("truth:",class_label_truth)
        # print(NN_ids)
        # print("predictions:",NN_names)
        index_in_pred = NN_names.index(class_label_truth) if class_label_truth in NN_names else -1
        # print("index:", index_in_pred)
        for name in NN_names:
            votes[name] += 1/(NN_names.index(name)+1) # antiproportional weighting
    if votes == {}:
        print("truth not in index")
        return -1

    votes = sorted(votes.items(),key=lambda x:x[1], reverse=True)
    # print(votes)
    # print("truth:",class_label_truth,"index:",[x[0] for x in votes].index(class_label_truth))
    prediction_class = votes[0][0]
    prediction = votes

    # classify sheet with index
    # prediction_class, prediction, _ = predict(descriptors, clf, truth=class_label_truth)
    
    # probabilities = list(zip(clf.classes_,prediction[0]))

    # probabilities.sort(key=lambda x: x[1], reverse=True)
    # print(probabilities)
    try:
        gt_index = [x[0] for x in prediction].index(class_label_truth)
        print("Prediction", prediction_class, "Truth at index", gt_index)
        # print("Prediction score %.3f Truth score %.3f" % (prediction[0][1], prediction[gt_index][1]))
        return gt_index
    except:
        print("truth not in index")
        return -1

def search_list(list_path, clf=None):
    # iterate over all sheets in list
    t0 = time()
    positions = []
    labels = []
    try:
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
                labels.append(class_label)
    except KeyboardInterrupt:
        pass
    finally:
        t1 = time()
        if len(positions) > 0:
            print("searching list took %f seconds. %f s per sheet" % (t1-t0,(t1-t0)/len(positions)))
            print(positions)
    return zip(labels,positions)

if __name__ == "__main__":
    if rebuild_index:
        clf = build_index(rsize=img_width_train)#restrict_class="524", restrict_range=200)


    # img_path = "E:/data/deutsches_reich/SBB/cut/SBB_IIIC_Kart_L 1330_Blatt 343_von_1925.tif"
    # ret = search_in_index(img_path, "343")
    # print(ret)
    # exit()

    # list_path = "E:/data/deutsches_reich/SBB/cut/list.txt"
    list_path = "E:/data/deutsches_reich/SLUB/cut/list_160_320.txt"
    lps = search_list(list_path)#, clf=clf)

    with open("index_result.csv","w") as fp:
        for l,p in lps:
            fp.write("%s : %d\n"%(l,p))

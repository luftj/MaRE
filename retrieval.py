import numpy as np
import cv2
import time
import logging
from operator import itemgetter
import joblib

import osm
import progressbar

import config
import indexing
import find_sheet
from eval_logs import mahalanobis_distance

def hist(ax, lbp):
    print("start hists")
    print(lbp.shape)
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                facecolor='0.5')

def detect_corners(image):
    from skimage.feature import corner_harris, corner_fast, corner_subpix, corner_peaks

    coords = corner_peaks(corner_fast(image), min_distance=10, threshold_rel=0)
    coords_subpix = None #corner_subpix(image, coords, window_size=13)
    return coords, coords_subpix

def match_template(image, template):
    from skimage import data
    from skimage.feature import match_template, peak_local_max
    import matplotlib.pyplot as plt

    result = match_template(image, template)
    # coordinates = corner_peaks(result, min_distance=10)
    # # print(coordinates[0:10])
    # scores = [result[y,x] for (y,x) in coordinates]
    # scores.sort(reverse=True)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    corr_coef = result[y,x]
    # print("max score",scores[0], scores[1], scores[1] <= 0.7 * scores[0])
    # if scores[1] <= 0.7 * scores[0]:
    #     plt.subplot("211")
    #     plt.imshow(template)
    #     plt.subplot("212")
    #     plt.imshow(image[y-30:y+30, x-30:x+30])
    #     # plt.show()
    # logging.info("best template matching score: %f" % corr_coef)
    return (x, y)

def plot_template(query_image, reference_image_border, template, x, y, match_x, match_y, pixel_high_percent, score):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(ncols=4)
    ax[0].imshow(query_image)
    htemplate, wtemplate = template.shape
    rect = plt.Rectangle((x-wtemplate//2, y-htemplate//2), wtemplate, htemplate, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)
    ax[1].imshow(template)
    ax[1].set_title("pixel %%: %f" % pixel_high_percent)
    match = reference_image_border[match_y:match_y+htemplate, match_x:match_x+wtemplate]
    ax[2].imshow(match)
    ax[3].imshow(reference_image_border, cmap=plt.cm.gray)
    ax[3].set_axis_off()
    ax[3].set_title('score: %f' % (score))
    # highlight matched region
    rect = plt.Rectangle((match_x, match_y), wtemplate, htemplate, edgecolor='r', facecolor='none')
    ax[3].add_patch(rect)
    plt.show()

def plot_template_matches(keypoints_q, keypoints_r, inliers,query_image, reference_image_border):
    import matplotlib.pyplot as plt
    from skimage.feature import plot_matches

    keypoints_q = np.fliplr(keypoints_q)
    keypoints_r = np.fliplr(keypoints_r)
    matches = np.array(list(zip(range(len(keypoints_q)),range(len(keypoints_r)))))

    print(f"Number of matches: {matches.shape[0]}")
    print(f"Number of inliers: {inliers.sum()}")
    fig, ax = plt.subplots(nrows=2, ncols=1)

    plot_matches(ax[0], (255-query_image), (255-reference_image_border), keypoints_q, keypoints_r,
                matches)#,alignment="vertical")
    plot_matches(ax[1], (255-query_image), (255-reference_image_border), keypoints_q, keypoints_r,
                matches[inliers])#,alignment="vertical")
    y =query_image.shape[0]
    plt.plot([500,1000,1000,500,500],[y,y,0,0,y],"r",linewidth=2)
    plt.plot([530,970,970,530,530],[y-30,y-30,30,30,y-30],"g",linewidth=1)
    # plt.xticks([],[])
    # plt.yticks([],[])
    # for spine in ax.spines:
    #     ax.spines[spine].set_visible(False)
    plt.show()

def template_matching(query_image, reference_image_border, window_size, patch_min_area=0.1, patch_max_area=0.8, plot=False):
    import dask

    keypoints_q = []
    keypoints_r = []
    # find interest points in query image (e.g. corners or white pixels)
    corners, corners_subpix = detect_corners(query_image)
    logging.debug("number of corners detected: %d" % len(corners))

    # height,width = query_image.shape
    # # reduce image size for performance with fixed aspect ratio. approx- same size as query, to make tempalte amtching work
    # reference_image = cv2.resize(reference_image, (width-window_size*2, height-window_size*2))
    # # reference_image = cv2.resize(reference_image, query_image.shape[::-1])

    # # make border of window size around reference image, to catch edge cases
    # reference_image_border = cv2.copyMakeBorder(reference_image, 
    #                                             window_size, window_size, window_size, window_size, 
    #                                             cv2.BORDER_CONSTANT, None, 0)
    # # reference_image_border = reference_image

    if plot:
        from matplotlib import pyplot as plt
        plt.subplot("121")
        plt.imshow(query_image)
        plt.scatter([x[1] for x in corners], [y[0] for y in corners], c="r", marker="x")
        plt.subplot("122")
        plt.imshow(reference_image_border)
        from skimage.feature import corner_harris, corner_fast, corner_subpix, corner_peaks
        ref_corners = corner_peaks(corner_fast(reference_image_border), min_distance=5)
        plt.scatter([x[1] for x in ref_corners], [y[0] for y in ref_corners], c="r", marker="x")
        # y =query_image.shape[0]
        # plt.plot([30,470,470,30,30], [y-30,y-30,30,30,y-30], "g", linewidth=1)
        plt.show()

    # match all sample points
    lazy_r = []
    for sample_point in corners:
        # sample interest point
        y,x = sample_point
        # extract template from query image around sampled point
        template = query_image[y-window_size:y+window_size, x-window_size:x+window_size]
        # skip patches that are not very descriptive
        num_pixels_high = cv2.countNonZero(template)
        pixel_high_percent = num_pixels_high / window_size**2

        if pixel_high_percent < patch_min_area or pixel_high_percent > patch_max_area:
            # don't consider ambiguous patches
            continue

        keypoints_q.append([x,y])
        
        # optional: reduce search space by only looking at/around interest points in reference image

        # find query template in reference image
        x = dask.delayed(match_template)(reference_image_border, template)
        lazy_r.append(x)

    
    results = dask.compute(*lazy_r)
    for x in results:
        match_x, match_y = x
        keypoints_r.append([match_x+window_size, match_y+window_size])
        # print("R,M:",(x,y),(match_x,match_y))
        # plot_template()

    # todo: optional: filter matches by score / lowe's test ratio

    # ransac those template matches!
    keypoints_q = np.array(keypoints_q)
    keypoints_r = np.array(keypoints_r)

    return keypoints_q, keypoints_r

def estimate_transform(keypoints_q, keypoints_r, query_image, reference_image_border, plot=False):
    from skimage.measure import ransac
    from skimage.transform import AffineTransform, SimilarityTransform
    logging.info("number of used keypoints: %d", len(keypoints_q))
    #logging.info("number of matched templates: %d", len(keypoints_r)) # all get matched
    
    if config.warp_mode_retrieval == "affine":
        warp_mode = AffineTransform
    elif config.warp_mode_retrieval == "similarity":
        warp_mode = SimilarityTransform

    if len(keypoints_r) <= 3:
        return 0, np.eye(3,3) # need to have enough samples for ransac.min_samples. For affine, at least 3

    model, inliers = ransac((keypoints_q, keypoints_r),
                        warp_mode, min_samples=3, stop_probability=config.ransac_stop_probability,
                        residual_threshold=5, max_trials=config.ransac_max_trials, random_state=config.ransac_random_state)

    if inliers is None:
        num_inliers = 0
    else:
        num_inliers = inliers.sum()

    # convert transform matrix to opencv format
    model = model.params
    model = np.linalg.inv(model)
    model = model.astype(np.float32) # opencv.warp doesn't take double

    if plot:
        plot_template_matches(keypoints_q,keypoints_r, inliers, query_image, reference_image_border)
        from skimage.transform import warp
        from matplotlib import pyplot as plt
        plt.subplot("131")
        plt.imshow(reference_image_border)
        plt.subplot("132")
        plt.imshow(query_image)
        plt.subplot("133")
        y = query_image.shape[0]
        plt.plot([30,470,470,30,30], [y-30,y-30,30,30,y-30], "g", linewidth=1)
        image1_warp = warp(query_image, model)
        plt.imshow(image1_warp)
        plt.show()

    return num_inliers, model

def feature_matching_kaze(query_image_small, reference_image_small):
    detector = cv2.KAZE_create(upright=True)
    # kp_query = detector.detect(query_image_small)
    # kp_reference = detector.detect(reference_image_small)
    # kp_query = sorted(kp_query, key=lambda x: -x.response)[:800]
    # kp_reference = sorted(kp_reference, key=lambda x: -x.response)[:800]
    # kps, dsc_q = detector.compute(query_image_small, kp_query)  # todo: use cv2.detectAndCompute instead, is faster
    # kps, dsc_r = detector.compute(reference_image_small, kp_reference)
    kp_query, dsc_q = detector.detectAndCompute(query_image_small, None)
    kp_reference, dsc_r = detector.detectAndCompute(reference_image_small, None)
    logging.info("#kps query %d" % len(kp_query))
    logging.info("#kps reference %d" % len(kp_query))

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(dsc_q, dsc_r)
    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)   # or pass empty dictionary
    # flann = cv2.FlannBasedMatcher(index_params,search_params)
    # knn_matches = flann.knnMatch(dsc_q, dsc_r, 2)
    # logging.info("#matches raw  %d" % len(knn_matches))
    # # Filter matches using Lowe's ratio test
    # ratio_thresh = 0.7
    # matches = []
    # for m,n in knn_matches:
    #     if m.distance < ratio_thresh * n.distance:
    #         matches.append(m)
    # logging.info("#matches refined  %d" % len(matches))
    
    keypoints_q = [kp_query[x.queryIdx].pt for x in matches]
    keypoints_r = [kp_reference[x.trainIdx].pt for x in matches]
    keypoints_q = np.array(keypoints_q)
    keypoints_r = np.array(keypoints_r)

    return keypoints_q, keypoints_r

def retrieve_best_match(query_image, bboxdict, processing_size):
    width, height = processing_size
    closest_image = None
    closest_bbox = None
    best_dist = -1

    start_time = time.time()

    score_list = []
    # $ python main.py /e/data/deutsches_reich/SBB/cut/list.txt data/blattschnitt_dr100_regular.geojson -r 20 --noimg
    # reduce image size for performance with fixed aspect ratio
    query_image_small = cv2.resize(query_image, (width,height))#, interpolation=cv2.INTER_AREA)

    progress = progressbar.ProgressBar(maxval=len(bboxdict))
    for sheet_name,bbox in progress(bboxdict.items()):
        idx = list(bboxdict.keys()).index(sheet_name)
        time_now = time.time()
        rivers_json = osm.get_from_osm(bbox)
        reference_river_image = osm.paint_features(rivers_json, bbox, img_size=[1000,850])

        # #--- kaze matching
        # # reduce image size for performance with fixed aspect ratio. approx- same size as query, to make tempalte amtching work
        # reference_image_small = cv2.resize(reference_river_image, (width, height), interpolation=cv2.INTER_AREA)
        # keypoints_q, keypoints_r = feature_matching_kaze(query_image_small, reference_image_small)
        # num_inliers, transform_model = estimate_transform(keypoints_q, keypoints_r, query_image_small, reference_image_small)

        #--- template matching
        window_size=config.template_window_size
        # reduce image size for performance with fixed aspect ratio. approx- same size as query, to make tempalte amtching work
        reference_image_small = cv2.resize(reference_river_image, (width-window_size*2, height-window_size*2), interpolation=cv2.INTER_AREA)

        # make border of window size around reference image, to catch edge cases
        reference_image_border = cv2.copyMakeBorder(reference_image_small, 
                                                    window_size, window_size, window_size, window_size, 
                                                    cv2.BORDER_CONSTANT, None, 0)
        keypoints_q, keypoints_r = template_matching(query_image_small, reference_image_border, window_size=window_size)
        num_inliers, transform_model = estimate_transform(keypoints_q, keypoints_r, query_image_small, reference_image_border)

        score_list.append((num_inliers, sheet_name))
        if closest_image is None or num_inliers > best_dist:
            closest_image = reference_river_image
            closest_bbox = bbox
            best_dist = num_inliers

        logging.info("target %d/%d Score %d Best %d, sheet: %s, bbox: %s, time: %f" % (idx+1, len(bboxdict), num_inliers, best_dist, sheet_name, bbox, time.time()-time_now))
    end_time = time.time()
    logging.info("total time spent: %f" % (end_time - start_time))
    score_list.sort(key=lambda x: x[0], reverse=True)
    return closest_image, closest_bbox, best_dist, score_list, transform_model

def retrieve_best_match_index(query_image, processing_size, sheets_path, restrict_number=100, truth=None, preload_reference=False):
    width, height = processing_size
    closest_bbox = None
    best_dist = -1
    score_list = []
    start_time = time.time()

    # reduce image size for performance with fixed aspect ratio
    query_image_small = cv2.resize(query_image, (width,height), interpolation=config.resizing_index_query)

    # extract features from query sheet
    keypoints, descriptors_query = indexing.extract_features(query_image_small, first_n=config.index_n_descriptors_query)
    descriptors_query = np.asarray(descriptors_query)

    if preload_reference:
        # load index from disk
        reference_descriptors = joblib.load(config.reference_descriptors_path)
        reference_keypoints = joblib.load(config.reference_keypoints_path)

    # if restrict number is 0, just return ground truth immediately, without spatial verification.
    # this speeds up testing registration only
    if restrict_number == 0 and truth:
        bbox = find_sheet.find_bbox_for_name(sheets_path, truth)
        bboxes = [bbox]
        sheet_predictions = [truth]
    else:
        # classify sheet with index
        print("Retrieving from index...")
        prediction = indexing.predict_annoy(descriptors_query)
        prediction = prediction[:restrict_number]
        score_cap = 1#0.4
        sheet_predictions, codebook_response = zip(*prediction)

        truth_index = sheet_predictions.index(truth) if truth in sheet_predictions else -1
        logging.info("Truth at position %d in index." % truth_index)
        print("Truth at position %d in index." % truth_index)
        logging.info("codebook response: %s" % (codebook_response,))
        ratios = [n1/n if n>0 else 0 for n,n1 in zip(codebook_response,codebook_response[1:])]
        logging.info("ratios: %s " % (ratios,))
        
        # don't to spatial verification if we have no chance of getting the correct prediction anyway
        if truth and (truth_index < 0 or truth_index > restrict_number):
            logging.info("verification pointless, skipping sheet")
            print("verification pointless, skipping sheet")
            return None, -1, [], None

        bboxes = find_sheet.get_ordered_bboxes_from_json(sheets_path, sheet_predictions)
    
    print("Verifying predictions...")
    progress = progressbar.ProgressBar(maxval=len(bboxes))
    for idx, bbox in progress(enumerate(bboxes)):
        sheet_name = sheet_predictions[idx]
        # if dict(prediction)[sheet_name] > score_cap:
        #     break
        time_now = time.time()
        
        # with precomputed descriptors
        if preload_reference:
            kp_reference = reference_keypoints[sheet_name]
            descriptors_reference = reference_descriptors[sheet_name]
        else:
            descriptors_reference = joblib.load(config.reference_descriptors_folder+"/%s.clf" % sheet_name)
            kp_reference = joblib.load(config.reference_keypoints_folder+"/%s.clf" % sheet_name)
        
        # Match descriptors.
        bf = cv2.BFMatcher(config.matching_norm, crossCheck=config.matching_crosscheck)
        matches = bf.match(np.asarray(descriptors_query), np.asarray(descriptors_reference)) # when providing tuples, opencv fails without warning, i.e. returns []
        keypoints_q = [keypoints[x.queryIdx].pt for x in matches]
        keypoints_r = [kp_reference[x.trainIdx] for x in matches]
        keypoints_r = [[x-config.index_border_train,y-config.index_border_train] for [x,y] in keypoints_r] # remove border from ref images, as they will not be there for registration
        keypoints_q = np.array(keypoints_q)
        keypoints_r = np.array(keypoints_r)
        
        num_inliers, transform_model = estimate_transform(keypoints_q, keypoints_r, None, None)

        score_list.append((num_inliers, sheet_name))
        if closest_bbox is None or num_inliers > best_dist:
            # this is the best match so far, keep for later
            closest_bbox = bbox
            best_dist = num_inliers
            best_sheet = sheet_name
            best_transform = transform_model

        maha = mahalanobis_distance([x[0] for x in score_list])
        logging.info("target %d/%d Sheet %s, Score %d Best %d, maha: %f, bbox: %s, time: %f" % (idx+1, len(bboxes), sheet_name, num_inliers, best_dist, maha, bbox, time.time()-time_now))
        # if idx>5 and maha >= 5.0:
        #     break # todo: should reflect how recent the change is, e.g. probability for better solution smaller than threshold, or maha didn't change for n sheets
        
        # early termination when correct sheet was already likely detected by unverified index rank
        if config.codebook_response_threshold and idx < len(bboxes)-1:
            if codebook_response[idx+1] > 0:
                test_ratio = codebook_response[idx]/codebook_response[idx+1]
            else:
                test_ratio = 0
            # logging.info("test ratio between this and next index: %0.2f" % test_ratio)
            # print("test ratio between this and next index: %0.2f" % test_ratio)
            if test_ratio > config.codebook_response_threshold:
                # if this sheet has a significantly higher CB response than the next, the remaining are probably just noise
                logging.info("breaking spatial verification because of testratio " + ("correctly" if truth_index <= idx else "wrongly"))
                print("breaking spatial verification because of testratio " + ("correctly" if truth_index <= idx else "wrongly"))
                break

    end_time = time.time()
    logging.info("total time spent for retrieval: %f" % (end_time - start_time))
    logging.info("avg time spent for retrieval: %f" % ((end_time - start_time)/len(bboxes)))
    score_list.sort(key=itemgetter(0), reverse=True)
    print("predicted sheet: %s" % best_sheet)

    # create a reference map image for the predicted location
    return closest_bbox, best_dist, score_list, best_transform
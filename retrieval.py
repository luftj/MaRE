import numpy as np
import cv2
import time
import logging
from operator import itemgetter
import joblib

import progressbar

import config
import indexing
import find_sheet
from eval_logs import mahalanobis_distance

def plot_matches(keypoints_q, keypoints_r, inliers, query_image, reference_image_border, bordersize=30):
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
    plt.plot([500+bordersize,1000-bordersize,1000-bordersize,500+bordersize,500+bordersize],[y-bordersize,y-bordersize,bordersize,bordersize,y-bordersize],"g",linewidth=1)
    plt.xticks([],[])
    plt.yticks([],[])
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    plt.show()

def estimate_transform(keypoints_q, keypoints_r, query_image, reference_image_border, plot=False):
    from skimage.measure import ransac
    from skimage.transform import AffineTransform, SimilarityTransform
    logging.info("number of used keypoints: %d" % len(keypoints_q))
    
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
        return 0, None
    else:
        num_inliers = inliers.sum()

    # convert transform matrix to opencv format
    model = model.params
    model = np.linalg.inv(model)
    model = model.astype(np.float32) # opencv.warp doesn't take double

    if plot:
        plot_matches(keypoints_q,keypoints_r, inliers, query_image, reference_image_border)
        from skimage.transform import warp
        from matplotlib import pyplot as plt
        plt.subplot(1,3,1)
        plt.imshow(reference_image_border)
        plt.subplot(1,3,2)
        plt.imshow(query_image)
        plt.subplot(1,3,3)
        y = query_image.shape[0]
        plt.plot([30,470,470,30,30], [y-30,y-30,30,30,y-30], "g", linewidth=1)
        image1_warp = warp(query_image, model)
        plt.imshow(image1_warp)
        plt.show()

    return num_inliers, model

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
        sheet_predictions = list(map(str,sheet_predictions)) # labels should always be strings

        truth_index = sheet_predictions.index(truth) if truth in sheet_predictions else -1
        logging.info("Truth at position %d in index." % truth_index)
        print("Truth at position %d in index." % truth_index)
        logging.info("codebook response: %s" % (codebook_response,))
        ratios = [n1/n if n>0 else 0 for n,n1 in zip(codebook_response,codebook_response[1:])]
        logging.info("ratios: %s " % (ratios,))
        
        # don't to spatial verification if we have no chance of getting the correct prediction anyway
        if config.skip_impossible_verification and truth and (truth_index < 0 or truth_index > restrict_number):
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

        if config.early_terminaten_heuristic == "maha_thresh":
            if idx>5 and maha >= 5.0:
                break # todo: should reflect how recent the change is, e.g. probability for better solution smaller than threshold, or maha didn't change for n sheets
        elif config.early_terminaten_heuristic== "codebook_response":
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
        elif not config.early_terminaten_heuristic:
            pass
        else:
            raise NotImplementedError(f"early termination heuristic {config.early_terminaten_heuristic} not implemented!")

    end_time = time.time()
    logging.info("total time spent for retrieval: %f" % (end_time - start_time))
    logging.info("avg time spent for retrieval: %f" % ((end_time - start_time)/len(bboxes)))
    score_list.sort(key=itemgetter(0), reverse=True)
    print("predicted sheet: %s" % best_sheet)

    # create a reference map image for the predicted location
    return closest_bbox, best_dist, score_list, best_transform
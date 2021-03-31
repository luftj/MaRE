import numpy as np
import cv2
import time
import logging

import osm
import progressbar

import config

def hist(ax, lbp):
    print("start hists")
    print(lbp.shape)
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                facecolor='0.5')

def lbp(image):
    from skimage.transform import rotate
    from skimage.feature import local_binary_pattern
    from skimage import data
    from skimage.color import label2rgb
    from matplotlib import pyplot as plt

    # settings for LBP
    radius = 3
    n_points = 8 * radius

    image = cv2.resize(image,(250,212))


    def overlay_labels(image, lbp, labels):
        mask = np.logical_or.reduce([lbp == each for each in labels])
        return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


    def highlight_bars(bars, indexes):
        for i in indexes:
            bars[i].set_facecolor('r')

    lbp = local_binary_pattern(image, n_points, radius)
    cv2.imshow("lbp",lbp)
    cv2.waitKey(30)

    # plot histograms of LBP of textures
    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    plt.gray()

    titles = ('edge', 'flat', 'corner')
    w = width = radius - 1
    edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    i_14 = n_points // 4            # 1/4th of the histogram
    i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
    corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                    list(range(i_34 - w, i_34 + w + 1)))

    label_sets = (edge_labels, flat_labels, corner_labels)

    for ax, labels in zip(ax_img, label_sets):
        ax.imshow(overlay_labels(image, lbp, labels))

    print("start plots")
    for ax, labels, name in zip(ax_hist, label_sets, titles):
        print("a plots")
        counts, _, bars = hist(ax, lbp)
        print("b plots")
        highlight_bars(bars, labels)
        ax.set_ylim(top=np.max(counts[:-1]))
        ax.set_xlim(right=n_points + 2)
        ax.set_title(name)

    ax_hist[0].set_ylabel('Percentage')
    for ax in ax_img:
        ax.axis('off')

def compute_hausdorff(query_image, reference_image, dist_fxn="cosine"):
    from hausdorff import hausdorff_distance

    query_points = cv2.findNonZero(query_image)
    reference_points = cv2.findNonZero(reference_image)
    if reference_points is None or len(reference_points) == 0:
        # empty image
        return [float("inf")]

    X = np.array([ p[0] for p in query_points])
    Y = np.array([ p[0] for p in reference_points])

    return  hausdorff_distance(Y, X, distance=dist_fxn)

def feature_matching_brief(img1, img2):
    from skimage import data
    from skimage import transform
    from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                                plot_matches, BRIEF)
    from skimage.color import rgb2gray
    import matplotlib.pyplot as plt


    img1 = cv2.resize(img1,(500,500))
    img2 = cv2.resize(img2,(500,500))


    keypoints1 = corner_peaks(corner_harris(img1), min_distance=5,
                            threshold_rel=0.1)
    keypoints2 = corner_peaks(corner_harris(img2), min_distance=5,
                            threshold_rel=0.1)

    extractor = BRIEF()

    extractor.extract(img1, keypoints1)
    keypoints1 = keypoints1[extractor.mask]
    descriptors1 = extractor.descriptors

    extractor.extract(img2, keypoints2)
    keypoints2 = keypoints2[extractor.mask]
    descriptors2 = extractor.descriptors


    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
    return len(matches12)
    
    # matched_keypoints_1 = keypoints1[matches12[:,0],:]
    # matched_keypoints_2 = keypoints2[matches12[:,1],:]

    # from skimage.transform import  AffineTransform,SimilarityTransform,EuclideanTransform

    # from skimage.measure import ransac

    # model_robust, inliers = ransac((matched_keypoints_1, matched_keypoints_2), AffineTransform, min_samples=3,
    #                             residual_threshold=0.2, max_trials=1000)
    # outliers = inliers == False
    # print(len(outliers), len(inliers))

    fig, ax = plt.subplots(nrows=1, ncols=1)

    plt.gray()

    plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12)
    ax.axis('off')
    ax.set_title("Query Image vs. Reference Image")

    # plt.show()

def compute_similarities(query_image, reference_image):
    # from skimage.metrics import structural_similarity as ssim
    # from skimage.metrics import mean_squared_error
    
    if cv2.countNonZero(query_image) == 0 or cv2.countNonZero(reference_image) == 0:
        # empty image
        return [0]
    
    query_image_resized = cv2.resize(query_image,reference_image.shape[::-1])
    n_matches = feature_matching_brief(query_image_resized,reference_image)
    
    # mse = mean_squared_error(query_image_resized, reference_image)
    # ssim_v = ssim(query_image_resized, reference_image, data_range=reference_image.max() - reference_image.min())
    return [n_matches]

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
    import cv2
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
    import cv2

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
        # sheet_name = sheet_names[idx]
        idx = list(bboxdict.keys()).index(sheet_name)
        time_now = time.time()
        rivers_json = osm.get_from_osm(bbox)
        reference_river_image = osm.paint_features(rivers_json, bbox)

        # #--- kaze matching
        # # reduce image size for performance with fixed aspect ratio. approx- same size as query, to make tempalte amtching work
        # reference_image_small = cv2.resize(reference_river_image, (width, height), interpolation=cv2.INTER_AREA)
        # keypoints_q, keypoints_r = feature_matching_kaze(query_image_small, reference_image_small)
        # num_inliers, transform_model = estimate_transform(keypoints_q, keypoints_r, query_image_small, reference_image_small)

        #--- template matching
        window_size=config.template_window_size
        # reduce image size for performance with fixed aspect ratio. approx- same size as query, to make tempalte amtching work
        reference_image_small = cv2.resize(reference_river_image, (width-window_size*2, height-window_size*2), interpolation=cv2.INTER_AREA)
        # reference_image = cv2.resize(reference_image, query_image.shape[::-1])

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

def retrieve_best_match_index(query_image, processing_size, sheets_path, restrict_number=100, truth=None):
    import joblib
    import indexing
    import find_sheet
    from eval_logs import mahalanobis_distance

    width, height = processing_size
    closest_image = None
    closest_bbox = None
    best_dist = -1

    start_time = time.time()

    score_list = []

    # todo: if restrict number < 0, just return ground truth. this would speed up testing registration only

    # reduce image size for performance with fixed aspect ratio
    query_image_small = cv2.resize(query_image, (width,height), interpolation=cv2.INTER_AREA)

    # extract features from query sheet
    keypoints, descriptors_query = indexing.extract_features(query_image_small, first_n=indexing.n_descriptors_query)
    # set up features as test set
    # load index from disk
    clf = joblib.load("index.clf")#"index_KAZE300.clf")
    sheetsdict = joblib.load("sheets.clf")#"index_KAZE300.clf")
    reference_keypoints = joblib.load("keypoints.clf")
    # classify sheet with index
    print("Retrieving from index...")
    # prediction_class, prediction, match_dict = indexing.predict(descriptors, clf)
    prediction_class, prediction, _ = indexing.predict_annoy(descriptors_query, sheetsdict, reference_keypoints=reference_keypoints)
    prediction=prediction[:restrict_number]
    score_cap = 1#0.4
    # print(prediction)
    sheet_predictions = [x[0] for x in prediction]
    # sheet_predictions = [x[0] for x in prediction if x[1] < score_cap]

    truth_index = sheet_predictions.index(truth) if truth in sheet_predictions else -1
    logging.info("Truth at position %d in index." % truth_index)
    print("Truth at position %d in index." % truth_index)
    test_ratio = prediction[0][1]/prediction[1][1]
    logging.info("test ratio between first two indices: %0.2f" % test_ratio)
    print("test ratio between first two indices: %0.2f" % test_ratio)

    # don't to spatial verification if we have no chance of getting the correct prediction anyway
    if truth and (truth_index < 0 or truth_index > restrict_number):
        logging.info("verification pointless, skipping sheet")
        print("verification pointless, skipping sheet")
        return None, None, -1, [], None

    # print(len(sheet_predictions))

    bboxes = find_sheet.get_ordered_bboxes_from_json(sheets_path, sheet_predictions)
    print("Verifying predictions...")
    progress = progressbar.ProgressBar(maxval=len(bboxes))
    for idx,bbox in progress(enumerate(bboxes)):
        sheet_name = sheet_predictions[idx]
        # if dict(prediction)[sheet_name] > score_cap:
        #     break
        time_now = time.time()

        # by redetecting
        # rivers_json = osm.get_from_osm(bbox)
        # reference_river_image = osm.paint_features(rivers_json, bbox)

        # #--- index matching
        # # reduce image size for performance with fixed aspect ratio
        # reference_image_small = cv2.resize(reference_river_image, (width, height))
        # reference_image_small = cv2.copyMakeBorder(reference_image_small, 
        #                                 indexing.border_train, indexing.border_train, indexing.border_train, indexing.border_train, 
        #                                 cv2.BORDER_CONSTANT, None, 0)
        # keypoints_q, keypoints_r = feature_matching_kaze(query_image_small, reference_image_small)
        
        # with precomputed descriptors
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # Match descriptors.
        matches = bf.match(descriptors_query, clf[sheet_name])
        keypoints_q = [keypoints[x.queryIdx].pt for x in matches]
        kp_reference = reference_keypoints[sheet_name]
        keypoints_r = [kp_reference[x.trainIdx] for x in matches]
        keypoints_r = [[x-indexing.border_train,y-indexing.border_train] for [x,y] in keypoints_r] # remove border from ref images, as they will not be there for registration
        keypoints_q = np.array(keypoints_q)
        keypoints_r = np.array(keypoints_r)
        
        # by querying annoy
        # # matches = match_dict[sheet_name]
        # from annoy import AnnoyIndex
        # from annoytest import get_kp_for_id
        # u = AnnoyIndex(64, indexing.annoydist)
        # u.load('index.ann') # super fast, will just mmap the file
        # keypoints_r = []
        # for desc in descriptors_query:
        #     nn = u.get_nns_by_vector(desc, 1)[0]
        #     keypoints_r.append(get_kp_for_id(clf,reference_keypoints,nn))
        # keypoints_q = [x.pt for x in keypoints]
        # keypoints_q = np.array(keypoints_q)
        # keypoints_r = np.array(keypoints_r)
        # print(keypoints_q)
        # print(keypoints_r)
        # print(len(keypoints_q),len(keypoints_r))
        num_inliers, transform_model = estimate_transform(keypoints_q, keypoints_r, None, None)

        score_list.append((num_inliers, sheet_name))
        if closest_bbox is None or num_inliers > best_dist:
            # closest_image = 1#reference_river_image
            closest_bbox = bbox
            best_dist = num_inliers
            best_sheet = sheet_name

        maha = mahalanobis_distance([x[0] for x in score_list])
        logging.info("target %d/%d Sheet %s, Score %d Best %d, maha: %f, bbox: %s, time: %f" % (idx+1, len(bboxes), sheet_name, num_inliers, best_dist, maha, bbox, time.time()-time_now))
        # if idx>5 and maha >= 5.0:
        #     break # todo: should reflect how recent the change is, e.g. probability for better solution smaller than threshold, or maha didn't change for n sheets
        
        # early termination when correct sheet was already likely detected by unverified index rank
        if test_ratio > 2:
            logging.info("breaking spatial verification because of testratio " + ("correctly" if truth_index==0 else "wrongly"))
            print("breaking spatial verification because of testratio " + ("correctly" if truth_index==0 else "wrongly"))
            break

    end_time = time.time()
    logging.info("total time spent: %f" % (end_time - start_time))
    logging.info("avg time spent: %f" % ((end_time - start_time)/len(bboxes)))
    score_list.sort(key=lambda x: x[0], reverse=True)
    # best_sheet = find_sheet.find_name_for_bbox(sheets_path, closest_bbox)
    print("predicted sheet: %s" % best_sheet)
    rivers_json = osm.get_from_osm(closest_bbox)
    closest_image = osm.paint_features(rivers_json, bbox)
    return closest_image, closest_bbox, best_dist, score_list, transform_model
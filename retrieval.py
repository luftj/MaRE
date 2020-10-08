import numpy as np
import cv2
import time
import logging

import osm
import progressbar

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
    coords_subpix = corner_subpix(image, coords, window_size=13)
    return coords, coords_subpix

def match_template(image, template):
    import matplotlib.pyplot as plt

    from skimage import data
    from skimage.feature import match_template

    result = match_template(image, template)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    corr_coef = result[y,x]
    return (x, y, corr_coef)

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

def plot_template_matches(keypoints_q,keypoints_r, inliers,query_image, reference_image_border):
    import matplotlib.pyplot as plt
    from skimage.feature import plot_matches

    matches = np.array(list(zip(range(len(keypoints_q)),range(len(keypoints_q)))))
    # inlier_keypoints_left = descs_q[matches[inliers, 0]]
    # inlier_keypoints_right = descs_r[matches[inliers, 1]]

    print(f"Number of matches: {matches.shape[0]}")
    print(f"Number of inliers: {inliers.sum()}")
    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    plot_matches(ax[0], (255-query_image), (255-reference_image_border), keypoints_q, keypoints_r,
                matches)
    plot_matches(ax[1], (255-query_image), (255-reference_image_border), keypoints_q, keypoints_r,
                matches[inliers])

    plt.show()

def match_templates(templates,reference_image_border, window_size):
    keypoints_r = []
    matching_score = 0
    for template in templates:      
        # optional: reduce search space by only looking at/around interest points in reference image
        # find query template in reference image
        match_x, match_y, score = match_template(reference_image_border, template)
        keypoints_r.append([match_y+window_size, match_x+window_size])
        # store best matching score
        matching_score += score
        
    # calculate metric from all matching scores for all samples. e.g. average
    matching_score /= len(keypoints_r)

    return keypoints_r, matching_score

def template_matching(query_image, reference_image, n_samples=50, window_size=30, patch_min_area=0.1, patch_max_area=0.8):
    import cv2
    import random
    from skimage.measure import ransac
    from skimage.transform import AffineTransform

    matching_score = 0
    keypoints_q = []
    keypoints_r = []

    # find interest points in query image (e.g. corners or white pixels)
    # white_pixels = list(cv2.findNonZero(query_image)) # returns np.array([x,y],...)
    # samples_positions = random.sample(white_pixels, k=n_samples)
    # samples_positions = np.array(samples_positions)

    # sample interest point
    corners, subpix = detect_corners(query_image)
    logging.info("number of corners detected: %d" % len(corners))

    # make border of window size around reference image, to catch edge cases
    reference_image_border = cv2.copyMakeBorder(reference_image, window_size,window_size,window_size,window_size, cv2.BORDER_CONSTANT, None, 0)
    
    scores = []
    # match all sample points
    for sample_point in corners:
        x,y = sample_point[0], sample_point[1]
        # extract template from query image around sampled point
        template = query_image[y-window_size:y+window_size, x-window_size:x+window_size]
        # skip patches that are not very descriptive
        num_pixels_high = cv2.countNonZero(template)
        pixel_high_percent = num_pixels_high / window_size**2

        if pixel_high_percent < patch_min_area or pixel_high_percent > patch_max_area:
            # don't consider ambiguous patches
            # newsample = corners[np.random.choice(corners.shape[0], 1, replace=False)]#[np.random.choice(list(corners), replace=False)]
            continue

        keypoints_q.append([y,x])
        
        # optional: reduce search space by only looking at/around interest points in reference image
        # find query template in reference image
        match_x, match_y, score = match_template(reference_image_border, template)
        keypoints_r.append([match_y+window_size, match_x+window_size])
        
        scores.append(score)
        # print("R,M,S:",(x,y),(match_x,match_y),score)
        # plot_template()

    # optional: filter matches by score

    # ransac those template matches!
    keypoints_q = np.array(keypoints_q)
    keypoints_r = np.array(keypoints_r)

    logging.info("number of matched templates: %d", len(keypoints_q))
    
    model, inliers = ransac((keypoints_q, keypoints_r),
                        AffineTransform, min_samples=3,
                        residual_threshold=5, max_trials=5000)

    if inliers is None:
        num_inliers = 0
        ransac_matching_score = 0
    else:
        num_inliers = inliers.sum()
        scores = np.array(scores)
        ransac_matching_score = sum(scores[inliers]) / num_inliers

    # plot_template_matches(keypoints_q,keypoints_r, inliers, query_image, reference_image_border)

    # store best matching score
    matching_score += score

    # calculate metric from all matching scores for all samples. e.g. average
    matching_score /= len(keypoints_r)

    return matching_score, ransac_matching_score, num_inliers

def retrieve_best_match(query_image, bboxes, processing_size):
    width, height = processing_size
    closest_image = None
    closest_bbox = None
    best_dist = -1

    start_time = time.time()

    score_list = []

    window_size = 30
    # reduce image size for performance with fixed aspect ratio
    query_image_small = cv2.resize(query_image, (width,height))

    progress = progressbar.ProgressBar(maxval=len(bboxes))
    for idx,bbox in progress(enumerate(bboxes)):
        time_now = time.time()
        rivers_json = osm.get_from_osm(bbox)
        reference_river_image = osm.paint_features(rivers_json, bbox)

        # reduce image size for performance with fixed aspect ratio. approx- same size as query, to make tempalte amtching work
        reference_image_small = cv2.resize(reference_river_image, (width-window_size*2,height-window_size*2))

        matching_score, ransac_matching_score, num_inliers = template_matching(query_image_small, reference_image_small, window_size=window_size)
        # matching_score, num_inliers = template_matching(keypoints_q,templates, query_image_small, reference_image_small)

        distances = [num_inliers, matching_score, ransac_matching_score]
        score_list.append((*distances, idx))
        if closest_image is None or distances[0] > best_dist:
            closest_image = reference_river_image
            closest_bbox = bbox
            best_dist = distances[0]

        logging.info("target %d/%d Score %d Best %d, bbox: %s, time: %f" % (idx+1, len(bboxes), distances[0], best_dist, bbox, time.time()-time_now))
    end_time = time.time()
    logging.info("total time spent: %f" % (end_time - start_time))
    score_list.sort(key=lambda x: x[0])
    return closest_image,closest_bbox,best_dist, score_list
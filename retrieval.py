import numpy as np
import cv2
import time

import osm

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

def retrieve_best_match(query_image, bboxes):
    closest_image = None
    closest_bbox = None
    best_dist = -1

    start_time = time.time()

    for idx,bbox in enumerate(bboxes):
        time_now = time.time()
        rivers_json = osm.get_from_osm(bbox)
        reference_river_image = osm.paint_features(rivers_json, bbox)
        
        # todo: roughly detect border in query image?
        # maybe it is not necessary with more robust matching technique
        # reference_river_image = cv2.copyMakeBorder(reference_river_image, 50,150,50,50, cv2.BORDER_CONSTANT, None, 0)

        distances = compute_similarities(query_image, reference_river_image)

        if closest_image is None or distances[0] > best_dist:
            closest_image = reference_river_image
            closest_bbox = bbox
            best_dist = distances[0]

        print("%d/%d" % (idx, len(bboxes)),"Distances:", *distances, bbox, time.time()-time_now)
        time_now = time.time()

    end_time = time.time()
    print("time spent:", end_time - start_time)

    return closest_image,closest_bbox,best_dist
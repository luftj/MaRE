import numpy as np
import cv2

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
import cv2
import numpy as np

def register_ECC(im2_gray, im1_gray, warp_mode = cv2.MOTION_AFFINE):
    print("starting registration...")
    # Find size of image1
    sz = im1_gray.shape

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 7000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-9

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

    return warp_matrix

def warp(im2_gray, warp_matrix, warp_mode = cv2.MOTION_AFFINE):
    sz = im2_gray.shape
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Show final results
    # cv2.imshow("Image 1", im1_gray)
    # cv2.imshow("Image 2", im2_gray)
    cv2.imshow("Aligned Image 2", im2_aligned)
    cv2.waitKey(0)
    return im2_aligned


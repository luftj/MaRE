import cv2
import numpy as np
import os

def register_ECC(query_image, reference_image, warp_mode = cv2.MOTION_AFFINE):
    print("starting registration...")
    # Find size of image1
    sz = reference_image.shape

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
    (cc, warp_matrix) = cv2.findTransformECC (reference_image,query_image,warp_matrix, warp_mode, criteria)

    return warp_matrix

def warp(image, warp_matrix, warp_mode = cv2.MOTION_AFFINE):
    sz = image.shape
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (image, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(image, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return im2_aligned

def georeference(inputfile, outputfile, bbox):
    left, top, right, bottom = (bbox[0], bbox[3], bbox[2], bbox[1])

    command = "gdal_translate -of GTiff -a_ullr %f %f %f %f -a_srs EPSG:4269 %s %s" % (left, top, right, bottom, inputfile, outputfile)
    print(command)
    os.system(command)

def align_map_image(map_image, query_image, reference_image):
    # register query and retrieved reference image for fine alignment
    query_image_small = cv2.resize(query_image,(500,500))
    reference_image_border = cv2.copyMakeBorder(reference_image, 150,150,150,150, cv2.BORDER_CONSTANT, None, 0)
    reference_image_small = cv2.resize(reference_image_border,(500,500))
    
    warp_matrix = register_ECC(query_image_small,reference_image_small)

    # query_aligned = registration.warp(query_image_small,warp_matrix)

    # todo: do this with the full sized input image
    map_img_small = cv2.resize(map_image,(500,500))
    map_img_aligned = warp(map_img_small,warp_matrix)
    
    # crop out border
    border_x = int(150 * reference_image_small.shape[1] / reference_image_border.shape[1])
    border_y = int(150 * reference_image_small.shape[0] / reference_image_border.shape[0])
    map_img_aligned = map_img_aligned[border_y:map_img_aligned.shape[0]-border_y, border_x:map_img_aligned.shape[1]-border_x]
    
    return map_img_aligned
import cv2
import numpy as np
import os
import logging
from time import time

def register_ECC(query_image, reference_image, warp_mode = cv2.MOTION_AFFINE):
    # taken from https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    
    logging.debug("starting registration...")
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
    time_start = time()

    left, top, right, bottom = (bbox[0], bbox[3], bbox[2], bbox[1])

    command = "gdal_translate -of GTiff -a_ullr %f %f %f %f -a_srs EPSG:4269 %s %s" % (left, top, right, bottom, inputfile, outputfile)
    logging.debug("gdal command: %s" % command)
    os.system(command)

    time_passed = time() - time_start
    logging.info("time: %f s for georeferencing" % time_passed)

def align_map_image(map_image, query_image, reference_image, target_size=(500,500)):
    time_start = time()
    
    print(target_size)
    logging.info("registration image resolution: %d,%d" % target_size)

    # register query and retrieved reference image for fine alignment
    query_image_small = cv2.resize(query_image, target_size, cv2.INTER_AREA)
    reference_image_border = cv2.copyMakeBorder(reference_image, 150,150,150,150, cv2.BORDER_CONSTANT, None, 0)
    reference_image_small = cv2.resize(reference_image_border, target_size, cv2.INTER_CUBIC)
    
    # get transformation matrix (map query=source to reference=target)
    warp_matrix = register_ECC(query_image_small, reference_image_small)

    # query_aligned = warp(query_image_small, warp_matrix)

    # convert affine parameters to homogeneous coordinates
    warp_matrix = np.vstack([warp_matrix, [0,0,1]])

    # scale by factor of target/original size
    scale_mat = np.eye(3,3,dtype=np.float32)
    scale_mat[0,0] *= map_image.shape[1] / target_size[0] 
    scale_mat[1,1] *= map_image.shape[0] / target_size[1] 
    scale_mat[2,2] *= 1

    warp_matrix = scale_mat @ warp_matrix @ np.linalg.inv(scale_mat) # complete transformation matrix
    warp_matrix = np.delete(warp_matrix, (2), axis=0) # drop homogeneous coordinates

    # do the warping with the full sized input image
    map_img_aligned = warp(map_image, warp_matrix)
    
    # crop out border
    border_x = int(150 * reference_image_small.shape[1] / reference_image_border.shape[1] * map_image.shape[1] / target_size[0])
    border_y = int(150 * reference_image_small.shape[0] / reference_image_border.shape[0] * map_image.shape[0] / target_size[1])
    map_img_aligned = map_img_aligned[border_y:map_img_aligned.shape[0]-border_y, border_x:map_img_aligned.shape[1]-border_x]
    
    time_passed = time() - time_start
    logging.info("time: %f s for registration" % time_passed)

    return map_img_aligned
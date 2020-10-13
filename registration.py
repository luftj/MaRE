import cv2
import numpy as np
import os
import logging
from time import time

import config

def register_ECC(query_image, reference_image, warp_matrix=None, warp_mode = cv2.MOTION_AFFINE):
    # taken from https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    
    logging.debug("starting registration...")
    # Find size of image1
    sz = reference_image.shape

    if warp_matrix is None:
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
    print(warp_matrix, warp_matrix.dtype)

    # Specify the number of iterations.
    number_of_iterations = 7000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-9

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(reference_image, query_image, warp_matrix, warp_mode, criteria)
    logging.info("found registration with score: %f" % cc)

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

def georeference(inputfile, outputfile, bbox, border=None):
    time_start = time()

    left, top, right, bottom = (bbox[0], bbox[3], bbox[2], bbox[1])

    if not border:
        command = "gdal_translate %s -a_ullr %f %f %f %f %s %s" % (config.gdal_output_options,left, top, right, bottom, inputfile, outputfile)
    else:
        command = "gdal_translate " + config.gdal_output_options + " "
        gcps = [
            (border[0],border[3],bbox[0],bbox[3]), # top left
            (border[0],border[1],bbox[0],bbox[1]), # bottom left
            (border[2],border[3],bbox[2],bbox[3]), # top right
            (border[2],border[1],bbox[2],bbox[1]), # bottom right
        ]
        for gcp in gcps:
            command += "-gcp %d %d %f %f " % gcp # pixel line easting northing
        command += inputfile + " " + outputfile# " map-with-gcps.tif"

    logging.debug("gdal command: %s" % command)
    os.system(command)

    time_passed = time() - time_start
    logging.info("time: %f s for georeferencing" % time_passed)

def align_map_image(map_image, query_image, reference_image, target_size=(500,500), crop=False, warp_matrix=None):
    time_start = time()

    logging.info("registration image resolution: %d,%d" % target_size)

    # register query and retrieved reference image for fine alignment
    query_image_small = cv2.resize(query_image, target_size, cv2.INTER_AREA)
    reference_image_border = cv2.copyMakeBorder(reference_image, 150,150,150,150, cv2.BORDER_CONSTANT, None, 0)
    reference_image_small = cv2.resize(reference_image_border, target_size, cv2.INTER_CUBIC)
    
    if config.warp_mode == "affine":
        warp_mode = cv2.MOTION_AFFINE
    elif config.warp_mode == "euclidean":
        warp_mode = cv2.MOTION_EUCLIDEAN

    # get transformation matrix (map query=source to reference=target)
    warp_matrix = register_ECC(query_image_small, reference_image_small, warp_matrix=warp_matrix, warp_mode=warp_mode)

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
    map_img_aligned = warp(map_image, warp_matrix, warp_mode=warp_mode)
    
    # crop out border
    border_x = int(150 * reference_image_small.shape[1] / reference_image_border.shape[1] * map_image.shape[1] / target_size[0])
    border_y = int(150 * reference_image_small.shape[0] / reference_image_border.shape[0] * map_image.shape[0] / target_size[1])
    if crop:
        map_img_aligned = map_img_aligned[border_y:map_img_aligned.shape[0]-border_y, border_x:map_img_aligned.shape[1]-border_x]
    
    time_passed = time() - time_start
    logging.info("time: %f s for registration" % time_passed)

    return map_img_aligned, (border_x, map_image.shape[0]-border_y, map_image.shape[1]-border_x, border_y)

def align_map_image_model(map_image, query_image, reference_image, warp_matrix, target_size=(500,500), crop=False):
    time_start = time()

    logging.info("registration image resolution: %d,%d" % target_size)

    # register query and retrieved reference image for fine alignment
    window_size=30
    width, height = target_size
    # map_image = cv2.resize(map_image,(width,height))
    reference_image_small = cv2.resize(reference_image, (width-window_size*2,height-window_size*2))
    reference_image_border = cv2.copyMakeBorder(reference_image_small, window_size,window_size,window_size,window_size, cv2.BORDER_CONSTANT, None, 0)
    warp_matrix = warp_matrix.params
    # scale by factor of target/original size
    scale_mat = np.eye(3,3,dtype=np.float32)
    scale_mat[0,0] *= map_image.shape[1] / (target_size[0])# - window_size*2) 
    scale_mat[1,1] *= map_image.shape[0] / (target_size[1])# - window_size*2)
    scale_mat[2,2] = 1

    warp_matrix = scale_mat @ warp_matrix @ np.linalg.inv(scale_mat) # complete transformation matrix
    warp_matrix = np.delete(warp_matrix, (2), axis=0) # drop homogeneous coordinates
    print(warp_matrix)

    # do the warping with the full sized input image
    map_img_aligned = warp(map_image, warp_matrix)
    
    # crop out border
    border_x = int(window_size * scale_mat[0,0])
    border_y = int(window_size * scale_mat[1,1])
    print("borders:",border_x, border_y)
    if crop:
        map_img_aligned = map_img_aligned[border_y:map_img_aligned.shape[0]-border_y, border_x:map_img_aligned.shape[1]-border_x]
    
    time_passed = time() - time_start
    logging.info("time: %f s for registration" % time_passed)

    return map_img_aligned, (border_x, map_image.shape[0]-border_y, map_image.shape[1]-border_x, border_y)
import cv2
import numpy as np
import os
import logging
from time import time
from pyproj import Transformer

import config

transform_sheet_to_out = Transformer.from_proj(config.proj_sheets, config.proj_out, always_xy=True) #, skip_equivalent=True # skip_equivalent is deprecated since some version

def register_ECC(query_image, reference_image, warp_matrix=None, warp_mode = cv2.MOTION_AFFINE, ret_cc=False):
    # adapted from https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    
    logging.debug("starting registration...")

    if warp_matrix is None:
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:
        if warp_mode != cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.delete(warp_matrix, (2), axis=0) # drop homogeneous coordinates

    # Define termination criteria: EPS or max iterations, whatever happens first
    termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                            config.registration_ecc_iterations,
                            config.registration_ecc_eps)

    # Run the ECC algorithm. The resulting transformation is stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(reference_image, query_image, warp_matrix, warp_mode, termination_criteria)
    logging.info("found registration with score: %f" % cc)

    if ret_cc:
        return warp_matrix, cc
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

def make_worldfile(inputfile, bbox, crop, target_size=None, input_size=None, warp=None):
    """ create a worldfile for a warped map image given bounding box GCPS
    bbox as [left_x, bottom_y, right_x, top_y]
    """

    # reproject the bounding box coordinates
    minxy = transform_sheet_to_out.transform(bbox[0], bbox[1]) # reproject lower left bbox corner
    maxxy = transform_sheet_to_out.transform(bbox[2], bbox[3]) # reproject upper right bbox corner
    bbox = minxy+maxxy

    # create border points from inv-warped reference border
    scale_mat = np.eye(3,3,dtype=np.float32)
    scale_mat[0,0] = input_size[1] / target_size[0] # x scaling factor
    scale_mat[1,1] = input_size[0] / target_size[1] # y scaling factor

    if warp is None:
        warp = np.identity(3)
    else:
        warp = np.vstack([warp,[0,0,1]])
    
    if crop:
        p0 = warp @ scale_mat @ np.array([
            0,
            0,
            1]) # upper left border point
        p1 = warp @ scale_mat @ np.array([
            0,
            target_size[1] - 2 * config.reference_map_padding,
            1]) # bottom left border point
        p2 = warp @ scale_mat @ np.array([
            target_size[0] - 2 * config.reference_map_padding,
            0,
            1]) # upper right border point
    else:
        p0 = warp @ scale_mat @ np.array([
            config.reference_map_padding,
            config.reference_map_padding,
            1]) # upper left border point
        p1 = warp @ scale_mat @ np.array([
            config.reference_map_padding,
            target_size[1]-config.reference_map_padding,
            1]) # bottom left border point
        p2 = warp @ scale_mat @ np.array([
            target_size[0]-config.reference_map_padding,
            config.reference_map_padding,
            1]) # upper right border point

    transform = cv2.getAffineTransform(
        np.vstack([p0[:2],p2[:2],p1[:2]]), 
        np.array([[bbox[0],bbox[3]],[bbox[2],bbox[3]],[bbox[0],bbox[1]]], dtype=np.float32))
    
    outputfile = os.path.splitext(inputfile)[0]+".wld"
    with open(outputfile,"w") as fw:
        fw.write("%.20f\n" % transform[0,0])
        fw.write("%.20f\n" % transform[1,0])
        fw.write("%.20f\n" % transform[0,1])
        fw.write("%.20f\n" % transform[1,1])
        fw.write("%.20f\n" % transform[0,2])
        fw.write("%.20f\n" % transform[1,2])
    
    logging.info("saved worldfile to: %s" % outputfile)
    return

def georeference(inputfile, outputfile, bbox, border=None):
    """use gdal to create a georeferenced output file.
    TO DO: use of border has been deprecated, needs to be changed to use border points/warp_matrix"""
    raise NotImplementedError("TO DO: use of border has been deprecated, needs to be changed to use border points/warp_matrix")
    time_start = time()
    minxy = transform_sheet_to_out.transform(bbox[0], bbox[1]) # reproject lower left bbox corner
    maxxy = transform_sheet_to_out.transform(bbox[2], bbox[3]) # reproject upper right bbox corner
    bbox = minxy+maxxy

    left, top, right, bottom = (bbox[0], bbox[3], bbox[2], bbox[1])

    if not border:
        command = "gdal_translate %s -a_ullr %f %f %f %f " % (config.gdal_output_options, left, top, right, bottom)
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
    
    command += ' "%s" "%s"' % (inputfile, outputfile) # " map-with-gcps.tif"
    logging.debug("gdal command: %s" % command)
    os.system(command)

    time_passed = time() - time_start
    logging.info("time: %f s for georeferencing" % time_passed)

def align_map_image(map_image, query_image, reference_image, target_size=(500,500), crop=False, transform_prior=None):
    time_start = time()

    logging.info("registration image resolution: %d,%d" % target_size)

    # register query and retrieved reference image for fine alignment
    query_image_small = cv2.resize(query_image, target_size, config.resizing_register_query)
    
    # we need some padding to make sure, we keep most of the map margins
    border_size = config.reference_map_padding
    reference_image = cv2.resize(reference_image, 
                                    (target_size[0] - border_size*2,
                                     target_size[1] - border_size*2),
                                    config.resizing_register_reference)
    reference_image_border = cv2.copyMakeBorder(reference_image, 
                                                border_size, border_size, border_size, border_size, 
                                                cv2.BORDER_CONSTANT, None, 0)
    
    if config.warp_mode_registration == "affine":
        warp_mode = cv2.MOTION_AFFINE
    elif config.warp_mode_registration == "euclidean":
        warp_mode = cv2.MOTION_EUCLIDEAN
    elif config.warp_mode_registration == "homography":
        warp_mode = cv2.MOTION_HOMOGRAPHY
    else:
        raise NotImplementedError("registration warp mode not supported:", config.warp_mode_registration)

    if not transform_prior is None:
        # when we pad the reference image (to keep map margins), the transform 
        # prior from RANSAC doesn't fit anymore. Adjust it with some algebra
        border_transform = np.eye(3,3,dtype=np.float32)
        border_transform[0,0] = reference_image_border.shape[1] / reference_image.shape[1]
        border_transform[1,1] = reference_image_border.shape[0] / reference_image.shape[0]
        border_transform[0,2] = -border_size
        border_transform[1,2] = -border_size
        transform_prior = border_transform @ transform_prior
    
    # get transformation matrix (map query=source to reference=target)
    warp_matrix = register_ECC(query_image_small, reference_image_border, warp_matrix=transform_prior, warp_mode=warp_mode)
    
    if config.warp_mode_registration != "homography":
        # convert affine parameters to homogeneous coordinates
        warp_matrix = np.vstack([warp_matrix, [0,0,1]])

    # scale by factor of target/original size: downscale original imput iamge to processing size
    scale_mat = np.eye(3,3,dtype=np.float32)
    scale_mat[0,0] *= map_image.shape[1] / target_size[0] # x scaling factor
    scale_mat[1,1] *= map_image.shape[0] / target_size[1] # y scaling factor

    warp_matrix = scale_mat @ warp_matrix @ np.linalg.inv(scale_mat) # complete transformation matrix in scale of original input map
    
    if config.warp_mode_registration != "homography":
        warp_matrix = np.delete(warp_matrix, (2), axis=0) # drop homogeneous coordinates

    # do the warping with the full sized input image
    map_img_aligned = warp(map_image, warp_matrix, warp_mode=warp_mode)
    
    # pixel coordinates of estimated map neatlines
    border_x = int(border_size * map_image.shape[0]/reference_image_border.shape[0])
    border_y = int(border_size * map_image.shape[1]/reference_image_border.shape[1])
    time_passed = time() - time_start
    logging.info("time: %f s for registration" % time_passed)

    if crop:
        # crop out border
        map_img_aligned = map_img_aligned[border_y:map_img_aligned.shape[0]-border_y, border_x:map_img_aligned.shape[1]-border_x]
    return map_img_aligned, warp_matrix

def align_map_image_model(map_image, query_image, reference_image, warp_matrix, target_size=(500,500), crop=False):
    time_start = time()

    logging.info("registration image resolution: %d,%d" % target_size)

    # register query and retrieved reference image for fine alignment
    # scale by factor of target/original size
    scale_mat = np.eye(3,3,dtype=np.float32)
    scale_mat[0,0] *= map_image.shape[1] / (target_size[0])# - window_size*2) 
    scale_mat[1,1] *= map_image.shape[0] / (target_size[1])# - window_size*2)
    scale_mat[2,2] = 1

    # corner points of ref image
    window_size = config.reference_map_padding
    upleft = np.array([window_size,window_size,1],dtype=np.float32)
    upleft_query = scale_mat @ ((warp_matrix) @ upleft)
    print("corner point",upleft,upleft_query)
    topright = np.array([target_size[0]-window_size,window_size,1],dtype=np.float32)
    topright_query = scale_mat @ ((warp_matrix) @ topright)
    print("corner point",topright,topright_query)
    botleft = np.array([window_size,target_size[1]-window_size,1],dtype=np.float32)
    botleft_query = scale_mat @ ((warp_matrix) @ botleft)
    print("corner point",botleft,botleft_query)
    botright = np.array([target_size[0]-window_size,target_size[1]-window_size,1],dtype=np.float32)
    botright_query = scale_mat @ ((warp_matrix) @ botright)
    print("corner point",botright,botright_query)

    window_size=0 
    upleft = np.array([window_size,window_size,1],dtype=np.float32)
    upleft_query = scale_mat @ ((warp_matrix) @ upleft)
    print("corner point UL",upleft,upleft_query)
    botright = np.array([target_size[0]-window_size,target_size[1]-window_size,1],dtype=np.float32)
    botright_query = scale_mat @ ((warp_matrix) @ botright)
    print("corner point BR",botright,botright_query)

    warp_matrix = scale_mat @ warp_matrix @ np.linalg.inv(scale_mat) # complete transformation matrix

    # do the warping with the full sized input image
    from skimage.transform import warp
    map_img_aligned = warp(map_image, warp_matrix, preserve_range=True)
    
    # pixel coordinates of estimated map neatlines
    border_left = int(upleft_query[0])
    border_right = int(botright_query[0])
    border_top = int(upleft_query[1])
    border_bot = int(botright_query[1])
    
    time_passed = time() - time_start
    logging.info("time: %f s for registration" % time_passed)

    if crop:
        # crop out border
        map_img_aligned = map_img_aligned[border_top:border_bot, border_left:border_right]

    warp_matrix = np.delete(warp_matrix, (2), axis=0) # drop homogeneous coordinates
    return map_img_aligned, warp_matrix
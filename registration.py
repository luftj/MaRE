import cv2
import numpy as np
import os
import logging
from time import time
from pyproj import Transformer, Proj

import config

transform_sheet_to_out = Transformer.from_proj(config.proj_sheets, config.proj_out, skip_equivalent=True, always_xy=True)

def register_ECC(query_image, reference_image, warp_matrix=None, warp_mode = cv2.MOTION_AFFINE):
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

def georeference_gcp(inputfile, outputfile, bbox, gcps):
    time_start = time()
    minxy = transform_sheet_to_out.transform(bbox[0], bbox[1]) # reproject lower left bbox corner
    maxxy = transform_sheet_to_out.transform(bbox[2], bbox[3]) # reproject upper right bbox corner
    # bb_br = transform_sheet_to_out.transform(bbox[2], bbox[1])
    # bb_ul = transform_sheet_to_out.transform(bbox[0], bbox[3])
    bbox = minxy+maxxy
    print(bbox)

    left, top, right, bottom = (bbox[0], bbox[3], bbox[2], bbox[1])
    bbox_corners = [ transform_sheet_to_out.transform(left, top),
                     transform_sheet_to_out.transform(left, bottom),
                     transform_sheet_to_out.transform(right, top),
                     transform_sheet_to_out.transform(right, bottom)]

    command = "gdal_translate " + config.gdal_output_options + " "

    for idx,gcp in enumerate(gcps):
        print("point",(*gcp[0:2], *(bbox_corners[idx])))
        command += "-gcp %f %f %f %f " % (*gcp[0:2], *(bbox_corners[idx]))  # pixel line easting northing
    command += inputfile + " " + outputfile# " map-with-gcps.tif"

    print(command)
    logging.debug("gdal command: %s" % command)
    os.system(command)

    time_passed = time() - time_start
    logging.info("time: %f s for georeferencing" % time_passed)

def make_worldfile(inputfile, bbox, border):
    """ create a worldfile for a warped map image given bounding box GCPS
    """
    minxy = transform_sheet_to_out.transform(bbox[0], bbox[1]) # reproject lower left bbox corner
    maxxy = transform_sheet_to_out.transform(bbox[2], bbox[3]) # reproject upper right bbox corner
    bbox = minxy+maxxy

    pixel_width = (bbox[2]-bbox[0])/(border[2]-border[0])
    pixel_height = (bbox[3]-bbox[1])/(border[3]-border[1])

    left_edge = bbox[0] - (border[0]-0.5) * pixel_width # subtract half pixel to get to the center of topleft corner
    top_edge = bbox[3] - (border[3]-0.5) * pixel_height # subtract half pixel to get to the center of topleft corner

    outputfile = os.path.splitext(inputfile)[0]+".wld"
    with open(outputfile,"w") as fw:
        fw.write("%.20f\n" % pixel_width)
        fw.write("0.0"+"\n")
        fw.write("0.0"+"\n")
        fw.write("%.20f\n" % pixel_height)
        fw.write("%.20f\n" % left_edge)
        fw.write("%.20f\n" % top_edge)
    
    logging.info("saved worldfile to: %s" % outputfile)

def georeference(inputfile, outputfile, bbox, border=None):
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
    query_image_small = cv2.resize(query_image, target_size, cv2.INTER_AREA)
    
    border_size = config.template_window_size

    reference_image = cv2.resize(reference_image, 
                                    (target_size[0] - border_size*2,
                                     target_size[1] - border_size*2),
                                    cv2.INTER_CUBIC)
                                    #cv2.INTER_AREA)
    reference_image_border = cv2.copyMakeBorder(reference_image, 
                                                border_size, border_size, border_size, border_size, 
                                                cv2.BORDER_CONSTANT, None, 0)
    reference_image_small = reference_image_border
    
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
    warp_matrix = register_ECC(query_image_small, reference_image_small, warp_matrix=transform_prior, warp_mode=warp_mode)
    # warp_matrix_small = warp_matrix
    if config.warp_mode_registration != "homography":
        # convert affine parameters to homogeneous coordinates
        warp_matrix = np.vstack([warp_matrix, [0,0,1]])

    # scale by factor of target/original size
    scale_mat = np.eye(3,3,dtype=np.float32)
    scale_mat[0,0] *= map_image.shape[1] / target_size[0] # x scaling factor
    scale_mat[1,1] *= map_image.shape[0] / target_size[1] # y scaling factor

    warp_matrix = scale_mat @ warp_matrix @ np.linalg.inv(scale_mat) # complete transformation matrix
    
    if config.warp_mode_registration != "homography":
        warp_matrix = np.delete(warp_matrix, (2), axis=0) # drop homogeneous coordinates

    # do the warping with the full sized input image
    map_img_aligned = warp(map_image, warp_matrix, warp_mode=warp_mode)
    
    # pixel coordinates of estimated map neatlines
    border_x = int(border_size * map_image.shape[0]/reference_image_border.shape[0])
    border_y = int(border_size * map_image.shape[1]/reference_image_border.shape[1])
    time_passed = time() - time_start
    logging.info("time: %f s for registration" % time_passed)

    # from matplotlib import pyplot as plt
    # border_x = border_y = config.template_window_size
    # ax = plt.subplot("231")
    # scale_m = np.eye(3,3,dtype=np.float32)
    # scale_m[0,0] = reference_image_border.shape[1] / reference_image.shape[1]
    # scale_m[1,1] = reference_image_border.shape[0] / reference_image.shape[0]
    # translate_mat = np.eye(3,3,dtype=np.float32)
    # translate_mat[0,2] = -border_size
    # translate_mat[1,2] = -border_size
    # ransac_transform_fixed = translate_mat @ scale_m @ transform_prior

    # # ransac_transform_full = scale_mat @ ransac_transform_fixed @ np.linalg.inv(scale_mat) # complete transformation matrix
    # ransac_transform_full = np.delete(ransac_transform_fixed, (2), axis=0)
    # ransac_aligned_fixed = warp(query_image_small, ransac_transform_full, warp_mode=warp_mode)
    # plt.imshow(ransac_aligned_fixed)
    # y = ransac_aligned_fixed.shape[0]
    # x = ransac_aligned_fixed.shape[1]
    # plt.plot([border_size,x-border_x,x-border_x,border_y,border_y], [y-border_y,y-border_y,border_x,border_x,y-border_y], "g", linewidth=1)
    # ax.set_title("RANSAC fixed")
    # print("ransac fixed")
    
    # ax = plt.subplot("232")
    # ecc_fixed = register_ECC(query_image_small, reference_image_small, warp_matrix=ransac_transform_fixed, warp_mode=warp_mode)
    # print("ecc fixed",ecc_fixed)
    # print(ransac_transform_fixed.shape,ecc_fixed.shape,scale_mat.shape)
    # # ecc_fixed = np.vstack([ecc_fixed, [0,0,1]])
    # # ecc_fixed = scale_mat @ ecc_fixed @ np.linalg.inv(scale_mat) # complete transformation matrix
    # # ecc_fixed = np.delete(ecc_fixed, (2), axis=0)
    # ecc_img_aligned_fixed = warp(query_image_small, ecc_fixed, warp_mode=warp_mode)
    # plt.imshow(ecc_img_aligned_fixed)
    # y = ecc_img_aligned_fixed.shape[0]
    # x = ecc_img_aligned_fixed.shape[1]
    # plt.plot([border_size,x-border_x,x-border_x,border_y,border_y], [y-border_y,y-border_y,border_x,border_x,y-border_y], "g", linewidth=1)
    # ax.set_title("ECC fixed")

    # ax = plt.subplot("233")
    # ax.set_title("target")
    # plt.imshow(reference_image_border)
    # y = reference_image_border.shape[0]
    # x = reference_image_border.shape[1]
    # plt.plot([border_size,x-border_x,x-border_x,border_y,border_y], [y-border_y,y-border_y,border_x,border_x,y-border_y], "g", linewidth=1)

    # ax = plt.subplot("234")
    # # ransac_transform = scale_mat @ transform_prior @ np.linalg.inv(scale_mat) # complete transformation matrix
    # ransac_transform = np.delete(transform_prior, (2), axis=0)
    # ransac_aligned = warp(query_image_small, ransac_transform, warp_mode=warp_mode)
    # plt.imshow(ransac_aligned)
    # print(ransac_aligned.shape[0])
    # y = ransac_aligned.shape[0]
    # x = ransac_aligned.shape[1]
    # plt.plot([border_size,x-border_x,x-border_x,border_y,border_y], [y-border_y,y-border_y,border_x,border_x,y-border_y], "g", linewidth=1)
    # ax.set_title("RANSAC")

    # ax = plt.subplot("235")
    # # warp_matrix_small = np.delete(warp_matrix_small, (2), axis=0)
    # ecc_img_aligned = warp(query_image_small, warp_matrix_small, warp_mode=warp_mode)
    # plt.imshow(ecc_img_aligned)
    # y = ecc_img_aligned.shape[0]
    # x = ecc_img_aligned.shape[1]
    # plt.plot([border_size,x-border_x,x-border_x,border_y,border_y], [y-border_y,y-border_y,border_x,border_x,y-border_y], "g", linewidth=1)
    # ax.set_title("ECC")
    # plt.show()

    if crop:
        # crop out border
        map_img_aligned = map_img_aligned[border_y:map_img_aligned.shape[0]-border_y, border_x:map_img_aligned.shape[1]-border_x]
        border = (0, map_img_aligned.shape[0], map_img_aligned.shape[1], 0)
    else:
        border = (border_x, map_image.shape[0]-border_y, map_image.shape[1]-border_x, border_y)
    return map_img_aligned, border

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
    window_size = config.template_window_size
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

    window_size=0 #template_window_size
    upleft = np.array([window_size,window_size,1],dtype=np.float32)
    upleft_query = scale_mat @ ((warp_matrix) @ upleft)
    print("corner point UL",upleft,upleft_query)
    botright = np.array([target_size[0]-window_size,target_size[1]-window_size,1],dtype=np.float32)
    botright_query = scale_mat @ ((warp_matrix) @ botright)
    print("corner point BR",botright,botright_query)

    warp_matrix = scale_mat @ warp_matrix @ np.linalg.inv(scale_mat) # complete transformation matrix
    # warp_matrix = np.delete(warp_matrix, (2), axis=0) # drop homogeneous coordinates

    # do the warping with the full sized input image
    from skimage.transform import warp
    map_img_aligned = warp(map_image, warp_matrix, preserve_range=True)
    # from matplotlib import pyplot as plt
    # plt.subplot("131")
    # plt.imshow(map_image)
    # plt.subplot("132")
    # plt.imshow(map_img_aligned)
    # plt.subplot("133")
    # plt.imshow(query_aligned)
    # plt.show()
    
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
        border= (0, map_img_aligned.shape[0], map_img_aligned.shape[1], 0)
    else:
        border = (border_left, border_bot, border_right, border_top)

    return map_img_aligned, border
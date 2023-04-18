
import numpy as np
import cv2

import config
import segmentation
import indexing
from retrieval import estimate_transform
from main import scale_proportional
from experiments.exp8a import get_query_image
from experiments.exp8a import get_reference_image

sheetfile = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
images_list = "E:/data/deutsches_reich/SLUB/cut/raw/list.txt"
query_sheet = "66"

# get input file
query_image_path = get_query_image(images_list, query_sheet)
print(query_image_path)
query_img = cv2.imdecode(np.fromfile(query_image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
# make segmentation of input file
query_mask = segmentation.extract_blue(query_img)
query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
# downscale to precessing size
processing_size = scale_proportional(query_img.shape, config.process_image_width)

query_img_small = cv2.resize(query_img, processing_size, config.resizing_register_query)
query_mask_small = cv2.resize(query_mask, processing_size, config.resizing_register_query)

keypoints_query, descriptors_query = indexing.extract_features(
                                                            query_mask_small, 
                                                            first_n=config.index_n_descriptors_query,
                                                            plot=False
                                                            )
reference_maps = ["11","66","183"]
for sheet in reference_maps:
    print("ref sheet:",sheet)

    reference_image = get_reference_image(sheet, sheetfile)
    border_size = config.reference_map_padding
    reference_image = cv2.resize(reference_image, 
                                    (processing_size[0] - border_size*2,
                                    processing_size[1] - border_size*2),
                                    config.resizing_register_reference)
    reference_image = cv2.copyMakeBorder(reference_image, 
                                                border_size, border_size*2, border_size, border_size, 
                                                cv2.BORDER_CONSTANT, None, 0)
    keypoints_reference, descriptors_reference = indexing.extract_features(
                                                            reference_image, 
                                                            first_n=config.index_n_descriptors_train,
                                                            plot=False
                                                            )
    
    # Match descriptors.
    bf = cv2.BFMatcher(config.matching_norm, crossCheck=config.matching_crosscheck)
    matches = bf.match(np.asarray(descriptors_query), np.asarray(descriptors_reference)) # when providing tuples, opencv fails without warning, i.e. returns []
    keypoints_q = [keypoints_query[x.queryIdx].pt for x in matches]
    keypoints_r = [keypoints_reference[x.trainIdx] for x in matches]
    # print (keypoints_r)
    # for k in keypoints_r:
    #     print(k.pt)
    keypoints_r = [[kp.pt[1]-config.index_border_train,kp.pt[0]-config.index_border_train] for kp in keypoints_r] # remove border from ref images, as they will not be there for registration
    keypoints_q = np.array(keypoints_q)
    keypoints_r = np.array(keypoints_r)

    num_inliers, transform_model = estimate_transform(
                                    keypoints_q, 
                                    keypoints_r, 
                                    query_mask_small, 
                                    reference_image,
                                    plot="matches")


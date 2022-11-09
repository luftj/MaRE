# exp 21: compare feature descriptor performance

# A: just show features on real query and reference image, count num matches pre and psot RANSAC
import experiments.config_e8a as config
from experiments.exp8a import get_query_image, get_reference_image, scale_proportional
import segmentation
import cv2
import numpy as np
import joblib
from retrieval import estimate_transform
import indexing
from matplotlib import pyplot as plt
from time import time

images_list = "E:/data/deutsches_reich/SLUB/cut/raw/list.txt"
# sheets_of_interest = ["248"]
sheets_of_interest = ["388","414"]#,"553","569","622","637"]
sheets_of_interest = ["414","553","569","622","637"]
sheets_of_interest = ["248","414","553","569","622","637"]
sheets_of_interest = ["248","414","553","569","622","637"]
sheets_of_interest = [str(x) for x in range(1,674,50)]
# descriptor_types = {"akaze_upright": cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT)}#indexing.detector_dict
descriptor_types = indexing.detector_dict
descriptor_types = {
    "kaze_upright": cv2.KAZE_create(upright=True),
    "akaze_upright": cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT),
    "surf_upright": cv2.xfeatures2d_SURF.create(upright=1),
    # "sift": cv2.SIFT.create(),
    # "ski_fast": Skimage_fast_detector(min_dist=5,thresh=0),
    # "cv_fast": cv2.FastFeatureDetector.create()
    }
outfile = "feature_comparison_A.csv"
results = []

for descriptor_name, descriptor_func in descriptor_types.items():
    print(descriptor_name)
    # set descript to use
    indexing.kp_detector = descriptor_func
    indexing.detector = descriptor_func
    
    for sheet in sheets_of_interest:
        print(sheet)
        query_image_path = get_query_image(images_list, sheet)
        if not query_image_path:
            continue
        print(query_image_path)
        query_map = cv2.imdecode(np.fromfile(query_image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # make segmentation of input file   
        query_img = segmentation.extract_blue(query_map)
        processing_size = scale_proportional(query_img.shape, config.process_image_width)
        query_img = cv2.resize(query_img, processing_size, config.resizing_register_query)

        # prepare reference image
        reference_image = get_reference_image(sheet)
        border_size = config.template_window_size
        reference_image = cv2.resize(reference_image, 
                                        (processing_size[0] - border_size*2,
                                        processing_size[1] - border_size*2),
                                        config.resizing_register_reference)
        reference_image = cv2.copyMakeBorder(reference_image, 
                                                    border_size, border_size, border_size, border_size, 
                                                    cv2.BORDER_CONSTANT, None, 0)

        # plt.subplot(1,2,1)
        # plt.imshow(query_img)
        # plt.subplot(1,2,2)
        # plt.imshow(reference_image)
        # plt.show()

        t0 = time()
        keypoints_query, descriptors_query = indexing.extract_features(
                                                        query_img, 
                                                        first_n=config.index_n_descriptors_query
                                                        )
        descriptors_query = np.asarray(descriptors_query)
        print(f"found {len(keypoints_query)} keypoints in query")
        # descriptors_reference = joblib.load(config.reference_descriptors_folder+"/%s.clf" % sheet)
        # keypoints_reference = joblib.load(config.reference_keypoints_folder+"/%s.clf" % sheet)
        keypoints_reference, descriptors_reference = indexing.extract_features(
                                                        reference_image, 
                                                        first_n=config.index_n_descriptors_query,#index_n_descriptors_train
                                                        )
        descriptors_reference = np.asarray(descriptors_reference)
        t1 = time()
        print(f"found {len(keypoints_reference)} keypoints in reference")
        
        # Match descriptors.
        bf = cv2.BFMatcher(config.matching_norm, crossCheck=config.matching_crosscheck)
        matches = bf.match(np.asarray(descriptors_query), np.asarray(descriptors_reference)) # when providing tuples, opencv fails without warning, i.e. returns []
        keypoints_q = [keypoints_query[x.queryIdx].pt for x in matches]
        # keypoints_r = [keypoints_reference[x.trainIdx] for x in matches]
        keypoints_r = [keypoints_reference[x.trainIdx].pt for x in matches]
        # keypoints_r = [[x-config.index_border_train,y-config.index_border_train] for [x,y] in keypoints_r] # remove border from ref images, as they will not be there for registration
        keypoints_q = np.array(keypoints_q)
        keypoints_r = np.array(keypoints_r)
        print(f"found {len(matches)} matching keypoints")

        num_inliers, transform_model = estimate_transform(
                                            keypoints_q, 
                                            keypoints_r, 
                                            query_img, 
                                            reference_image, 
                                            plot=False) # True
        print(f"found {num_inliers} matches after RANSAC")
        print(f"used {t1-t0} seconds")
        results.append({
            "descriptor": descriptor_name,
            "sheet": sheet,
            "num_kp_query": len(keypoints_query),
            "num_kp_reference": len(keypoints_reference),
            "num_matches": len(matches),
            "num_inliers": num_inliers,
            "time": t1-t0
        })

print(results)
with open(outfile,"w") as fw:
    fw.write(",".join(results[0].keys()))
    fw.write("\n")
    for row in results:
        fw.write(",".join(map(str,row.values())))
        fw.write("\n")

for descriptor_name in descriptor_types.keys():
    print(f"""avg num matches for {descriptor_name}: {sum([x["num_matches"] for x in results if x["descriptor"] == descriptor_name])/len(sheets_of_interest)}""")
    print(f"""avg num inliers for {descriptor_name}: {sum([x["num_inliers"] for x in results if x["descriptor"] == descriptor_name])/len(sheets_of_interest)}""")
    print(f"""avg time for {descriptor_name}: {sum([x["time"] for x in results if x["descriptor"] == descriptor_name])/len(sheets_of_interest)}""")

# B: show performance for whole synthetic dataset: index ranks and full retrieval

# C: show performance for real dataset: index ranks and full retrieval
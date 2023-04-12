import numpy as np
import cv2
from experiments.exp8a import scale_proportional
import indexing
import glob
from matplotlib import pyplot as plt

# colors = [(0,100,255)]
#for image_path in glob.glob("figures/reference_samples/*-inv.png"):
for image_path in glob.glob("figures/066*_inv_1500.png"):
    # image_path = "figures/reference_samples/11-inv.png"
    print(image_path)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    process_image_width = 500
    processing_size = scale_proportional(image.shape, process_image_width)
    image = cv2.resize(image, processing_size, cv2.INTER_AREA)

    index_n_descriptors=300
    keypoints_query, descriptors_query = indexing.extract_features(
                                                            image, 
                                                            first_n=index_n_descriptors,
                                                            plot=True
                                                            )
    outpath = image_path.replace(".png","_kps.png")
    # plt.savefig(outpath,bbox_inches='tight')
    plt.close()

    for idx,descriptor in enumerate(descriptors_query[::10]):
        mat = np.reshape(descriptor, (8,8))
        plt.imshow(mat)
        plt.gray()
        plt.axis('off')
        outpath2 = outpath.replace("_kps",f"_desc{idx}")
        print(outpath2)
        plt.savefig(outpath2, bbox_inches='tight')
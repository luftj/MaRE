
import os
import glob
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2
import numpy as np

from config import path_output
import segmentation

list_path = "E:/data/deutsches_reich/wiki/highres/list.txt"

list_dir = os.path.dirname(list_path) + "/"

sheets = {}

sheet_names = ["529","12","3","4","5",
                "6","7","8","9","10-18",
                "11",#"79a",
                "168","173","259",
                "327","328","329","330","331",
                "332","333","352","353","354",
                "355","356","357","358","377",
                "378","379","381","382","383",
                "402","403","404","405","406",
                "407","408","428","429","430",
                "431","432","433","434","571",
                "572","581","602"]
error_results = [127.425356,  808.127598, 405.070386, 349.664682, 509.326544, 
                 1977.204952, 499.567129, 333.353408, 114.199929, 347.203024, 
                 164.871981, # 18179.592263, 
                 191.158490,  75.028294, 52.330538, 
                 102.093746,  121.107682,  2067.600979, 92.453029, 1543.557139, 
                 91.655742,   51.111890,   129.926327, 87.675772, 171.246463, 
                 61.944273,   77.858277,   58.707810, 275.575773, 107.694777, 
                 1601.318240, 84.639471,   126.455419, 87.476919, 119.161954, 
                 1690.922109, 1239.491436, 62.393875, 126.555694, 99.391196, 
                 77.121876,   100.562299,  140.500505, 89.768985, 69.173359, 
                 64.417055,   73.200442,   83.135307, 64.073825, 1424.003588, 
                 85.241935,   82.453438,   73.992250]
segs={"12": [1.25, 0.90, 1.39],"1-2": [1.85, 1.32, 1.40],"3": [2.34, 2.78, 0.84],"4": [4.91, 2.10, 2.34],"5": [3.61, 5.87, 0.62],"6": [0.84, 1.83, 0.46],"7": [3.40, 5.58, 0.61],"8": [3.24, 3.94, 0.82],"9": [2.00, 1.80, 1.11],"10-18": [2.73, 2.64, 1.03],"11": [1.73, 4.90, 0.35],"168": [3.10, 4.09, 0.76],"173": [4.30, 1.48, 2.90],"259": [4.40, 2.83, 1.56],"327": [4.68, 2.66, 1.76],"328": [3.33, 2.19, 1.52],"329": [3.03, 1.40, 2.17],"330": [3.71, 1.26, 2.94],"331": [2.99, 1.00, 2.98],"332": [3.50, 1.39, 2.52],"333": [3.62, 0.90, 4.00],"352": [3.84, 2.01, 1.91],"353": [4.32, 3.31, 1.30],"354": [2.92, 1.50, 1.94],"355": [4.05, 1.51, 2.69],"356": [2.97, 1.35, 2.19],"357": [3.32, 2.15, 1.54],"358": [3.84, 0.47, 8.24],"377": [3.81, 1.34, 2.84],"378": [3.23, 2.77, 1.17],"379": [4.66, 1.39, 3.35],"380": [6.39, 2.09, 3.05],"381": [3.88, 2.73, 1.42],"382": [4.83, 2.05, 2.35],"383": [4.44, 1.02, 4.37],"402": [3.77, 1.91, 1.98],"403": [3.93, 1.74, 2.25],"404": [4.61, 2.70, 1.70],"405": [3.49, 2.15, 1.62],"406": [4.03, 1.66, 2.43],"407": [4.23, 1.78, 2.38],"408": [5.45, 3.17, 1.72],"428": [4.93, 1.07, 4.60],"429": [3.23, 1.27, 2.54],"430": [3.57, 5.29, 0.67],"431": [5.03, 1.90, 2.65],"432": [5.33, 1.55, 3.43],"433": [4.28, 1.19, 3.60],"434": [4.49, 2.76, 1.62],"571": [1.71, 1.29, 1.32],"572": [1.94, 3.18, 0.61],"581": [3.63, 1.93, 1.88],"602": [2.38, 2.70, 0.88],"79a": [0.49, 0.20, 2.52],"529": [1.10, 1.85, 0.59]}

# x=[segs[s][1]/segs[s][0] for s in sheet_names]
# x=[segs[s][2] for s in sheet_names]
x=[segs[s][0] for s in sheet_names]
y=[segs[s][1] for s in sheet_names]

plt.scatter(x,y,label="factor",c=error_results,cmap="Blues")
# x=[segs[s][1]-segs[s][0]*0.5 for s in sheet_names]
# y=error_results
# plt.scatter(x,y,label="error")
# x=[segs[s][0] for s in sheet_names]
# plt.scatter(x,y,c="g",label="map")
# x=[segs[s][1] for s in sheet_names]
# plt.scatter(x,y,c="y",label="ref")
plt.legend()
# plt.xlabel("ref-map")
plt.xlabel("blue map")#/ref")
plt.ylabel("ref map")#error [m]")

# for err in y:
    # idx=error_results.index(err)
    # plt.annotate(sheet_names[idx],(segs[sheet_names[idx]][2], err))
for s in sheet_names:
    plt.annotate(s,(segs[s][0], segs[s][1]))
plt.plot([0,6],[0,6],"r--")

# for err in y:
#     idx=error_results.index(err)
#     plt.annotate(sheet_names[idx],(segs[sheet_names[idx]][1], err))

# for err in y:
#     idx=error_results.index(err)
#     plt.annotate(sheet_names[idx],(segs[sheet_names[idx]][0], err))

plt.show()

exit()

with open(list_path, encoding="utf-8") as list_file:
    for line in list_file:
        line = line.strip()
        if not "," in line:
            print("skipping line: no ground truth given %s" % line)
            continue
        img_path, ground_truth = line.split(",")
        if not os.path.isabs(img_path[0]):
            img_path = os.path.join(list_dir,img_path)

        # print(img_path, ground_truth)
        ref_path = os.path.join(path_output,"refimg_%s_*" % ground_truth)
        ref_path =  list(glob.iglob(ref_path))[0]
        print(ground_truth, ref_path)

        sheets[ground_truth] = { "path": img_path,
                                "ref_path": ref_path}

for sheet in sheets.keys():
    img_path = sheets[sheet]["path"]
    ref_path = sheets[sheet]["ref_path"]
    # map_img = imread(img_path)
    map_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    water_mask = segmentation.extract_blue(map_img, 5) # extract rivers
    ref_img = cv2.imdecode(np.fromfile(ref_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    num_blue_pixels_map = cv2.countNonZero(water_mask)
    percent_blue_pixels_map = num_blue_pixels_map / (water_mask.shape[0] * water_mask.shape[1]) * 100

    num_blue_pixels_ref = cv2.countNonZero(ref_img)
    percent_blue_pixels_ref = num_blue_pixels_ref / (ref_img.shape[0] * ref_img.shape[1]) * 100

    print("sheet-%%map-%%ref-factor. %s: %.2f %.2f %.2f" % (sheet, percent_blue_pixels_map, percent_blue_pixels_ref, percent_blue_pixels_map/percent_blue_pixels_ref))

    # plt.subplot(1,2,1)
    # plt.imshow(map_img)
    # plt.subplot(1,2,2)
    # plt.gray()
    # plt.imshow(water_mask)
    # plt.show()
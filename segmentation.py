import cv2
from simple_cb import simplest_cb

import logging

def extract_blue(img, cb_percent):
    # img_cb = simplest_cb(img, args.percent)
    img_cie = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    img_cie = simplest_cb(img_cie, cb_percent)

    # TODO: adjust kernel sizes to image resolution
    ksize = (5, 5) 
    img_cie = cv2.blur(img_cie, ksize)  

    l,a,b = cv2.split(img_cie)

    # cv2.imshow("b",cv2.resize(b,(b.shape[1]//2,b.shape[0]//2)))

    lowerBound = (10, 0, 0)
    upperBound = (255, 90, 100) #(255, 90, 80) # (255, 90, 70)

    img_thresh = cv2.inRange(img_cie, lowerBound, upperBound)

    num_blue_pixels = cv2.countNonZero(img_thresh)
    percent_blue_pixels = num_blue_pixels / (img_thresh.shape[0] * img_thresh.shape[1]) * 100
    logging.info("segmented %d pixels, %.2f percent" % (num_blue_pixels, percent_blue_pixels))

    # retm,b_threshold = cv2.threshold(b,5,255,cv2.THRESH_BINARY)
    # cv2.imshow("b_threshold",cv2.resize(b_threshold,(b_threshold.shape[1]//2,b_threshold.shape[0]//2)))
    # retm,a_threshold = cv2.threshold(a,5,255,cv2.THRESH_BINARY)
    # cv2.imshow("a_threshold",cv2.resize(a_threshold,(a_threshold.shape[1]//2,a_threshold.shape[0]//2)))

    # ksize = (3, 3) 
    # img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_ERODE, ksize)
    # import matplotlib.pyplot as plt
    # import numpy as np
    # print(img[:,:,2].shape)
    # rgb_img = np.stack([img[:,:,2],img[:,:,1],img[:,:,0]],axis=-1)
    # plt.subplot(2,2,1), plt.imshow(rgb_img)
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,2), plt.imshow(img_cie)
    # plt.title('LAB Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,3), plt.imshow(img_thresh)
    # plt.title('thresh Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,4)
    # plt.hist((b.ravel(),a.ravel()), 256)
    # plt.title('b Histogram')#, plt.xticks([]), plt.yticks([])
    # plt.show()

    # ksize = (5, 5) 
    # opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, ksize)
    # # cv2.imshow("opening",cv2.resize(opening,(opening.shape[1]//2,opening.shape[0]//2)))

    # ksize = (33, 33) 
    # # closing = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, ksize)
    # # cv2.imshow("closing",cv2.resize(closing,(closing.shape[1]//2,closing.shape[0]//2)))

    # dilation = cv2.dilate(img_thresh,ksize,iterations = 4)
    # cv2.imshow("dilation",cv2.resize(dilation,(dilation.shape[1]//2,dilation.shape[0]//2)))

    return img_thresh


if __name__ == "__main__":
    import argparse
    from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path string")
    parser.add_argument("--percent", help="colour balancethreshold", default=5, type=int)
    args = parser.parse_args()
    
    map_img = cv2.imread(args.input) # load map image
    
    plt.subplot(1, 2, 1)
    map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    plt.imshow(map_img_rgb)
    plt.subplot(1, 2, 2)
    segmented_image = extract_blue(map_img, args.percent)
    plt.gray()
    plt.imshow(segmented_image)
    plt.title("segmented map image")
    plt.show()
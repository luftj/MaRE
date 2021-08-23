import cv2
from simple_cb import simplest_cb, better_cb
import random
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

import logging
import config

def plot_lab_3d(img_cie, rgb_img):
    n_points = 5000
    pp = np.reshape(img_cie,(img_cie.shape[0]*img_cie.shape[1],3))
    rgbpp = list(np.reshape(rgb_img,(rgb_img.shape[0]*rgb_img.shape[1],3)))
    pp = list(pp)
    # print(img_cie.shape,pp.shape)
    # print(pp[0:3])
    points_i = random.sample(range(len(pp)),n_points)
    points = [pp[i] for i in points_i]#random.sample(pp, n_points)
    # print(points[0:3])
    x = [int(p[2]) for p in points]# if p[0]>10] # b
    y = [int(p[1]) for p in points]# if p[0]>10] # a
    z = [int(p[0]) for p in points]# if p[0]>10] # a
    c = [rgbpp[i]/256 for i in points_i]# if pp[i][0]>10]
    # print(c[0:3])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, c=c)
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_zlabel("L")
    plt.show()

def plot_lab_2d(rgb_img, img_cie):
    n_points = 10000
    pp = np.reshape(img_cie,(img_cie.shape[0]*img_cie.shape[1],3))
    rgbpp = list(np.reshape(rgb_img,(rgb_img.shape[0]*rgb_img.shape[1],3)))
    pp = list(pp)
    # print(img_cie.shape,pp.shape)
    # print(pp[0:3])
    points_i = random.sample(range(len(pp)),n_points)
    points = [pp[i] for i in points_i]#random.sample(pp, n_points)
    # print(points[0:3])
    x = [int(p[2]) for p in points if p[0]>10] # b
    y = [int(p[1]) for p in points if p[0]>10] # a
    c = [rgbpp[i]/256 for i in points_i if pp[i][0]>10]
    # print(c[0:3])
    
    plt.scatter(x,y, c=c)
    plt.xlabel("b")
    plt.ylabel("a")
    plt.show()

def plot_hsv_2d(bgr_img, img_hsv):
    n_points = 100000
    pp = np.reshape(img_hsv,(img_hsv.shape[0]*img_hsv.shape[1],3))
    rgbpp = list(np.reshape(bgr_img,(bgr_img.shape[0]*bgr_img.shape[1],3)))
    pp = list(pp)
    points_i = random.sample(range(len(pp)),n_points)
    points = [pp[i] for i in points_i]#random.sample(pp, n_points)
    # print(points[0:3])
    x = [int(p[0]) for p in points if p[1]>30] # h
    z = [int(p[1]) for p in points if p[1]>30] # s
    y = [int(p[2]) for p in points if p[1]>30] # v
    c = [rgbpp[i][::-1]/256 for i in points_i if pp[i][1]>30] # original colours of each sampled point converted to float RGB
    # print(c[0:3])
    
    plt.scatter(x,y, c=c)
    plt.xlabel("H")
    plt.ylabel("V")
    plt.show()

def extract_blue(img, plot=False):
    """Perform colour-based segmentation of an image.

    Keyword arguments:
    img -- opencv compatible image (BGR)
    
    returns: the segmented image as binary matrix
    """
    if config.segmentation_colourspace == "lab":
        img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif config.segmentation_colourspace == "hsv":
        if img.shape[-1] == 4:
            print("has alpha!", img.shape)
            exit()
        img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        raise NotImplementedError("colour space %s not implemented!" % config.segmentation_colourspace)

    img_converted = better_cb(img_converted, config.segmentation_colourbalance_percent)

    # adjust kernel sizes to image resolution
    ksize = config.segmentation_blurkernel
    if ksize[0] > 1: # only blur with sensible kernel
        img_converted = cv2.blur(img_converted, ksize)  

    lowerBound = config.segmentation_lowerbound
    upperBound = config.segmentation_upperbound

    img_thresh = cv2.inRange(img_converted, lowerBound, upperBound)

    num_blue_pixels = cv2.countNonZero(img_thresh)
    percent_blue_pixels = num_blue_pixels / (img_thresh.shape[0] * img_thresh.shape[1]) * 100
    logging.info("segmented %d pixels, %.2f percent" % (num_blue_pixels, percent_blue_pixels))

    if plot:
        # print(img[:,:,2].shape)
        # # rgb_img = np.stack([img[:,:,2],img[:,:,1],img[:,:,0]],axis=-1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2,2,1)
        plt.imshow(rgb_img)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,2)
        plt.imshow(img_converted)
        plt.title('cvt Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,3)
        plt.imshow(img_thresh)
        plt.title('thresh Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,4)
        a,b,c = cv2.split(img_converted)
        if config.segmentation_colourspace == "lab":
            plt.hist((c.ravel(),b.ravel(),a.ravel()), 256, color=["b","r","k"], label=["b","a","L"])
        elif config.segmentation_colourspace == "hsv":
            plt.hist((c.ravel(),b.ravel(),a.ravel()), 256, color=["b","r","k"], label=["V","S","H"])
        plt.title('Histogram')#, plt.xticks([]), plt.yticks([])
        plt.legend()
        plt.show()

        if config.segmentation_colourspace == "lab":
            plot_lab_2d(img, img_converted)
        elif config.segmentation_colourspace == "hsv":
            plot_hsv_2d(img, img_converted)
        else:
            print("colour space plot not implemented")

    ksize = config.segmentation_openingkernel
    if ksize[0] > 1: # only open when a sensible kernel is set
        opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, ksize)
        img_thresh = opening
    
    ksize = config.segmentation_closingkernel
    if ksize[0] > 1: # only open when a sensible kernel is set
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, config.segmentation_closingkernel)

    return img_thresh#, img_converted

def load_and_run(map_path):
    map_img = cv2.imread(map_path) # load map image
    
    segmented_image = extract_blue(map_img, plot=False)
    return segmented_image

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path string")
    parser.add_argument("--plot", help="show segmented image", action="store_true")
    parser.add_argument("--save", help="save segmented image", action="store_true")
    args = parser.parse_args()
    
    # segmented_image = load_and_run(args.input)
    
    map_img = cv2.imread(args.input) # load map image # todo: allow utf8 filenames
    segmented_image = extract_blue(map_img, plot=False)

    if args.plot:
        from matplotlib import pyplot as plt
        ax1 = plt.subplot(1, 3, 1)
        map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
        ax1.imshow(map_img_rgb)
        ax3 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        map_img_hsv = cv2.cvtColor(map_img, cv2.COLOR_BGR2HSV)
        ax3.imshow(map_img_hsv)
        ax2 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
        plt.gray()
        ax2.imshow(segmented_image)
        # segmented_image,map_img_hsv = extract_blue(map_img, plot=False)
        # ax1 = plt.subplot(1, 3, 1)
        # ax1.imshow(map_img_rgb)
        # ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        # ax2.imshow(map_img_hsv)
        # ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
        # plt.gray()
        # ax3.imshow(segmented_image)
        plt.title("segmented map image")
        plt.show()
    if args.save:
        import os
        cv2.imwrite(os.path.splitext(args.input)[0]+"_mask.png", segmented_image)
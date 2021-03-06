# taken from
# https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
# http://www.morethantechnical.com/2015/01/14/simplest-color-balance-with-opencv-wcode/
# http://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html
# all credit goes to the original authors
# this code file is not licensed for distribution

import cv2
import math
import numpy as np
import sys
import argparse
import logging

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb_gray(img, percent):
    assert(len(img.shape) == 2)
    assert(percent > 0 and percent < 100)
    logging.debug("Colour balance: treshold %.2f %%" % (percent))

    half_percent = percent / 200.0

    assert(len(img.shape) == 2)
    # find the low and high precentile values (based on the input percentile)
    height, width = img.shape
    vec_size = width * height
    flat = img.reshape(vec_size)

    assert(len(flat.shape) == 1)

    flat = np.sort(flat)

    n_cols = flat.shape[0]

    low_val  = flat[math.floor(n_cols * half_percent)]
    high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

    logging.debug("Colour balance: Lowval %d Highval %d" % (low_val, high_val))

    # saturate below the low percentile and above the high percentile
    thresholded = apply_threshold(img, low_val, high_val)
    # scale the channel
    normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)

    return normalized

def simplest_cb(img, percent):
    assert(img.shape[2] == 3)
    if not (percent > 0 and percent < 100):
        return img
    logging.debug("Colour balance: treshold %.2f %%" % (percent))

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert(len(channel.shape) == 2)
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert(len(flat.shape) == 1)

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        logging.debug("Colour balance: Lowval %d Highval %d" % (low_val, high_val))

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

def better_cb(img, percent=1):
    """Apply Simplest Color Balance algorithm
    Reimplemented based on https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
    thanks to https://gist.github.com/JackDesBwa"""

    if not percent or percent == 0 or percent == 100:
        return img
    
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path string")
    parser.add_argument("output", help="output file path string")
    parser.add_argument("percent", help="threshold", default=5,type=int)
    args = parser.parse_args()

    img = cv2.imread(args.input)
    out = simplest_cb(img, args.percent)
    cv2.imwrite(args.output, out)
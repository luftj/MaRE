import os
import argparse
import numpy as np
from skimage import measure
import skimage.io
import skimage.filters
from matplotlib import pyplot as plt


def get_dominant_colour(image, plot=False):
    # find dominant colour by histogram analysis

    R,G,B = np.split(image, 3, axis=2) # split channels

    R_hist = np.histogram(R.ravel(), bins=256)[0]
    G_hist = np.histogram(G.ravel(), bins=256)[0]
    B_hist = np.histogram(B.ravel(), bins=256)[0]

    max_R = np.argmax(R_hist)
    max_G = np.argmax(G_hist)
    max_B = np.argmax(B_hist)
    print("dominant colour (RGB):", max_R, max_G, max_B)

    if plot:
        plt.bar(range(256),R_hist, color="r")
        plt.bar(range(256),G_hist, color="g")
        plt.bar(range(256),B_hist, color="b")
        plt.axvline(max_R, color='orange')
        plt.axvline(max_G, color='orange')
        plt.axvline(max_B, color='orange')
        plt.show()
    return max_R, max_G, max_B

def threshold_image(image, dominant_colour, interval = 50, plot=False):
    max_R, max_G, max_B = dominant_colour

    thresh_image = np.ones(image.shape)
    thresh_image[image[:,:,0] < (max_R - interval)] = 0
    thresh_image[image[:,:,0] > (max_R + interval)] = 0
    thresh_image[image[:,:,1] < (max_G - interval)] = 0
    thresh_image[image[:,:,1] > (max_G + interval)] = 0
    thresh_image[image[:,:,2] < (max_B - interval)] = 0
    thresh_image[image[:,:,2] > (max_B + interval)] = 0
    thresh_image = thresh_image[:,:,0]

    blur = skimage.filters.gaussian(thresh_image, sigma=(5, 5), multichannel=True)

    if plot:
        plt.subplot("121")
        plt.imshow(thresh_image)
        plt.subplot("122")
        plt.imshow(blur)
        plt.show()

    return blur

def find_largest_contour(thresh_image):
    print("find contours")
    contours = measure.find_contours(thresh_image, 0.5)
    print(len(contours), "contours found")

    print("finding largest contour...")
    box = None
    max_size = -1
    for contour in contours:
        x = contour[:,1]
        y = contour[:,0]
        size = (max(x) - min(x)) * (max(y) - min(y))
        if size > max_size:
            box = contour
            max_size = size
    return box

def crop_image(image, percent_margin=0.01, plot=False):
    thresh_image = threshold_image(image, get_dominant_colour(image, plot), plot=plot)

    box = find_largest_contour(thresh_image)

    print("crop")
    y = box[:,1]
    x = box[:,0]

    min_x, max_x, min_y, max_y = int(min(x)), int(max(x)), int(min(y)),int(max(y))
    margin_x = int((max_x-min_x)*percent_margin)
    margin_y = int((max_y-min_y)*percent_margin)
    print(min_x, max_x, min_y, max_y, margin_x, margin_y)
    cropped = image[min_x+margin_x:max_x-margin_x, min_y+margin_y:max_y-margin_y]

    if plot:
        print("plot")
        plt.subplot("121")
        plt.gray()
        plt.imshow(thresh_image)
        plt.plot(box[:, 1], box[:, 0], linewidth=2)
        plt.subplot("122")
        plt.imshow(cropped)
        plt.show()
    return cropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="input directory path string for images to be cropped")
    parser.add_argument("--plot", help="set this to true to show debugging plots", action="store_true")
    args = parser.parse_args()

    in_dir = args.indir
    file_types = [".tif", ".png"]
    out_dir = os.path.join(in_dir, "cut")
    os.makedirs(out_dir, exist_ok=True)

    for file_name in [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]:
        if not file_name[-4:] in file_types:
            print("skipping %s..." % file_name)
            continue
        image_path = os.path.join(in_dir, file_name)
        print("processing image at", image_path)
        image = skimage.io.imread(image_path)

        cropped_image = crop_image(image, plot=args.plot)

        print("saving cropped image to disk...")
        out_path = os.path.join(out_dir, file_name)
        skimage.io.imsave(out_path, cropped_image)

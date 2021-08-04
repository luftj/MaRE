import argparse
import osm
import progressbar
import find_sheet
import cv2
import numpy as np
import os

def noisy(noise_typ,image):
    """Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.1
        amount = 0.554
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)        
        noisy = image + image * gauss
        return noisy

def degrade_circles(image, num_circles):
    import random
    # radii = [5,25]
    radii = [15,50]
    out_img = img
    for n in range(num_circles):
        out_img = cv2.circle(out_img,
                            (random.randint(0,image.shape[0]),random.randint(0,image.shape[1])),
                            random.randint(*radii), 
                            0, thickness=-1)
    return out_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sheets", help="sheets json file path string")
    parser.add_argument("list", help="path to image list")
    parser.add_argument("output", help="output dir")
    parser.add_argument("--wld", help="make worldfiles too", action="store_true", default=False)
    parser.add_argument("--circles", help="degrade with circles", type=int, default=0)
    args = parser.parse_args()

    # logging.basicConfig(filename='logs/osmkdr500.log', level=logging.DEBUG) # gimme all your loggin'!
    progress = progressbar.ProgressBar()
    # sheets_file = "data/blattschnitt_dr100_regular.geojson"
    # sheets_file = "E:/data/deutsches_reich/Blattschnitt/blattschnitt_dr100_merged.geojson"
    # sheets_file = "E:/data/dr500/blattschnitt_kdr500_wgs84.geojson"
    # bboxes = find_sheet.get_bboxes_from_json(sheets_file)
    bbox_dict = find_sheet.get_dict(args.sheets, True)
    import os
    os.makedirs(args.output, exist_ok=True)

    sheets = []
    with open(args.list, encoding="utf-8") as fr:
        for line in fr:
            _, sheet = line.strip().split(",")
            sheets.append(sheet)

    bboxes = [bbox_dict[x] for x in sheets]
    for idx, bbox in enumerate(progress(bboxes)):
        gj = osm.get_from_osm(bbox)
        img = osm.paint_features(gj,bbox)

        # make margin
        margin_px = 100
        img = cv2.copyMakeBorder(img, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=0)

        # todo: make noise
        # img = noisy("s&p", img)
        if args.circles > 0:
            img = degrade_circles(img, args.circles)

        # cv2.imshow("output",img)
        # cv2.waitKey(-1)
        # exit()
        
        outpath = args.output + "/%s.png" % sheets[idx]
        cv2.imwrite(outpath, img)

        if args.wld:
            # georef this to allow easy check
            from registration import make_worldfile
            make_worldfile(outpath, bbox,[0,img.shape[0],img.shape[1],0])

    with open(args.output+"/list.txt","w") as fw:
        for s in sheets:
            fw.write("%s.png,%s\n" % (s,s) )
import joblib
from time import time

def extract_features(image, first_n=None):
    import cv2, numpy as np
    detector = cv2.BRISK_create()
    # first_n = None
    # detector = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT, diffusivity=cv2.KAZE_DIFF_CHARBONNIER)
    # detector = cv2.KAZE_create(upright=True)
    if first_n:
        kps = detector.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:first_n]
        kps, dsc = detector.compute(image, kps)  # todo: use cv2.detectAndCompute instead, is faster
    else:
        kps, dsc = detector.detectAndCompute(image, None)
    return dsc

def resize_by_width(shape, new_width):
    width = new_width
    f = width / shape[1]
    height = int(f * shape[0])
    return (width, height)

def restrict_bboxes(sheets_path, target, num):
    import find_sheet
    # for debugging: don't check against all possible sheet locations, but only a subset
    bboxes = find_sheet.get_bboxes_from_json(sheets_path)
    idx = find_sheet.get_index_of_sheet(sheets_path, target)
    if idx is None:
        raise ValueError("ground truth name %s not found" % target)
    print("restricted bbox %s by %d around idx %s" %(target, num, idx))
    start = max(0, idx-(num//2))
    return bboxes[start:idx+(num//2)+1]

def build_index(rsize=None, restrict_class=None, restrict_range=None):
    import progressbar
    import cv2
    from sklearn.ensemble import RandomForestClassifier

    import find_sheet, osm
    t0 = time()

    sheets_path = "data/blattschnitt_dr100_regular.geojson"
    if restrict_class and restrict_range:
        bboxes = restrict_bboxes(sheets_path, restrict_class, restrict_range)
    else:
        bboxes = find_sheet.get_bboxes_from_json(sheets_path)

    X = []
    Y = []

    index_dict = {}
    
    progress = progressbar.ProgressBar(maxval=len(bboxes))
    for bbox in progress(bboxes):
        rivers_json = osm.get_from_osm(bbox)
        reference_river_image = osm.paint_features(rivers_json, bbox)

        # reduce image size for performance with fixed aspect ratio
        processing_size = resize_by_width(reference_river_image.shape, rsize if rsize else 500)
        reference_image_small = cv2.resize(reference_river_image, processing_size)
        class_label = find_sheet.find_name_for_bbox(sheets_path, bbox)
        if not class_label:
            print("error in class name. skipping bbox", bbox)
            continue

        # extract features of sheet
        try:
            descriptors = extract_features(reference_image_small, first_n=300)
        except Exception as e:
            print(e)
            print("error in descriptors. skipping sheet", class_label)
            continue
        if descriptors is None or descriptors[0] is None:
            print("\n\nerror in descriptors. skipping sheet", class_label,"\n\n")
            continue
        # get class label
        # add features and class=sheet to index
        # X.extend(descriptors)
        # Y.extend([class_label]*len(descriptors))
        index_dict[class_label] = descriptors
    # clf = RandomForestClassifier(n_estimators=100, verbose=True, n_jobs=-1)
    # clf = clf.fit(X, Y)
    
    t1 = time()
    print("building index took %f seconds. %f s per sheet" % (t1-t0,(t1-t0)/len(bboxes)))

    # save index to disk
    joblib.dump(index_dict, "index.clf", compress=3)
    print("compress and store time: %f s" % (time()-t1))
    # return clf
    return index_dict

def predict(sample, clf):
    import cv2
    # prediction_class = clf.predict(sample)
    # prediction = clf.predict_proba(sample)
    prediction = []
    for label in clf.keys():
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # Match descriptors.
        matches = bf.match(sample, clf[label])
        # matches = bf.match(clf[label], sample)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        matches = matches[:100]

        ## Lowe's test ratio refinement
        # nn_matches = bf.knnMatch(sample, clf[label], 2)
        # matches = []
        # nn_match_ratio = 0.7 # Nearest neighbor matching ratio
        # for m,n in nn_matches:
        #     if m.distance < nn_match_ratio * n.distance:
        #         matches.append(m)

        # nn_matches_reverse = bf.knnMatch(clf[label], sample, 2)
        # for m,n in nn_matches_reverse:
        #     if m.distance < nn_match_ratio * n.distance:
        #         matches.append(m)

        # matches2 = bf.match(clf[label], sample)
        # matches2 = sorted(matches2, key = lambda x:x.distance)
        # matches2 = matches2[:100]
        # matches.extend(matches2)
        distances = [x.distance for x in matches]
        # print(distances)
        # score = len(matches)
        if len(matches) == 0:
            prediction.append((label,1))
        else:
            score = sum(distances)/len(matches)
            prediction.append((label,score))
    prediction.sort(key=lambda x: x[1])#, reverse=True)
    # print(prediction)
    prediction_class = prediction[0][0]
    return prediction_class, prediction


def search_in_index(img_path, class_label_truth, cb_percent=5, rsize=None, clf=None):
    import os
    import cv2, numpy as np
    import segmentation

    # load query sheet
    map_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # seegment query sheet
    water_mask = segmentation.extract_blue(map_img, cb_percent) # extract rivers
    # image size for intermediate processing
    processing_size = resize_by_width(map_img.shape, rsize if rsize else 500)
    water_mask_small = cv2.resize(water_mask, processing_size)
    # extract features from query sheet
    descriptors = extract_features(water_mask_small, first_n=300)
    # set up features as test set
    # load index from disk
    if not clf:
        clf = joblib.load("index.clf")  
    # classify sheet with index
    prediction_class, prediction = predict(descriptors, clf)
    print(prediction_class)
    
    # probabilities = list(zip(clf.classes_,prediction[0]))

    # probabilities.sort(key=lambda x: x[1], reverse=True)
    # print(probabilities)
    try:
        gt_index = [x[0] for x in prediction].index(class_label_truth)
        print("Truth at index", gt_index)
        return gt_index
    except:
        print("truth not in index")
        return -1

def search_list(list_path, clf=None):
    # iterate over all sheets in list
    import os
    t0 = time()
    positions = []
    with open(list_path, encoding="utf-8") as list_file:
        for line in list_file:
            line = line.strip()
            print(line)
            if not "," in line:
                print("skipping line: no ground truth given %s" % line)
                continue
            img_path, class_label = line.split(",")
            if not os.path.isabs(img_path[0]):
                list_dir = os.path.dirname(list_path) + "/"
                img_path = os.path.join(list_dir,img_path)
            pos = search_in_index(img_path, class_label, clf=clf)
            positions.append(pos)
            
    t1 = time()
    print("searching list took %f seconds. %f s per sheet" % (t1-t0,(t1-t0)/len(positions)))
    return positions

if __name__ == "__main__":
    clf = build_index()#restrict_class="524", restrict_range=200)

    # img_path = "E:/data/deutsches_reich/SBB/cut/SBB_IIIC_Kart_L 1330_Blatt 669 von 1908.tif"#SBB_IIIC_Kart_L 1330_Blatt 259 von 1925_koloriert.tif"
    # class_label_truth = "669"
    # search_in_index(img_path, class_label_truth)

    list_path = "E:/data/deutsches_reich/SBB/cut/list.txt"
    positions = search_list(list_path)#, clf=clf)
    print(positions)
    # positions200 = [352, 414, 2, 6, 0, 2, 3, 0, 2, 155, 229, 104, 326, 20, 2, 4,  69, 36, 0, 5, 2, 2, 3, 33, 28, 0, 8, 2, 326, 2, 3, 2, 3]
    # positions300 = [399, 311, 1, 2, 0, 3, 5, 0, 4,  68, 180, 117, 467,  3, 2, 4, 125, 28, 0, 6, 2, 2, 6, 38, 33, 0, 3, 2, 227, 2, 3, 2, 3]
    # positions400 = [432, 192, 1, 3, 0, 3, 5, 2, 4,  81, 166, 185, 435,  4, 2, 4, 125, 84, 0, 4, 1, 3, 26, 60, 33, 0, 3, 12, 230, 2, 3, 2, 3]
    maha = [3.92,4.17,63.62,18.6,37.51,35.58,15.36,14.71,24.17,15.68,11.55,7.6,4.52,10.98,24.36,25.53,6.96,11.7,28.92,36.52,37.75,21.36,25.05,4.76,11.95,8.6,15.76,11.58,2.7,42.41,40.9,53.08,13.34]
    from matplotlib import pyplot as plt
    plt.boxplot([x for x in positions if x >= 0])
    # plt.scatter(maha, positions300, c="r")
    # plt.scatter(maha, positions200, c="b")
    # plt.scatter(maha, positions400, c="g")
    # plt.show()

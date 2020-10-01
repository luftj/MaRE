import json

def make_test_bboxes(ref_bbox):
    bbox = ref_bbox
    bbox_size = (bbox[2]-bbox[0],bbox[3]-bbox[1])

    bboxes = []
    for x in (0,1,2,3):
        for y in (-2,-1,0):
            nbox = [bbox[0]+x*bbox_size[0],
                    bbox[1]+y*bbox_size[1],
                    bbox[2]+x*bbox_size[0],
                    bbox[3]+y*bbox_size[1]]
            bboxes.append(nbox)
    print(*bboxes,sep="\n")
    return bboxes

def get_bboxes_from_json(filepath):
    bboxes = []

    with open(filepath) as file:
        json_data = json.load(file)

        for feature in json_data["features"]:
            if all(i in feature["properties"] for i in("left","right","bottom","top")):
                minx = feature["properties"]["left"]
                maxx = feature["properties"]["right"]
                miny = feature["properties"]["bottom"]
                maxy = feature["properties"]["top"]
            else:
                minx = min([p[0] for p in feature["geometry"]["coordinates"][0]])
                maxx = max([p[0] for p in feature["geometry"]["coordinates"][0]])
                miny = min([p[1] for p in feature["geometry"]["coordinates"][0]])
                maxy = max([p[1] for p in feature["geometry"]["coordinates"][0]])

            bbox = [minx, miny, maxx, maxy]
            bboxes.append(bbox)

    return bboxes

def get_index_of_sheet(sheetfile, sheet):
    with open(sheetfile) as file:
        json_data = json.load(file)

        for idx,feature in enumerate(json_data["features"]):
            if feature["properties"]["blatt_100"] == sheet:
                return idx
            if feature["properties"]["blatt_ostmark"] == sheet:
                return idx
            if feature["properties"]["blatt_polen"] == sheet:
                return idx
    
    return None # not found

def find_name_for_bbox(sheetfile, bbox):
    with open(sheetfile) as file:
        json_data = json.load(file)

        for feature in json_data["features"]:
            minx, miny, maxx, maxy = bbox
            if feature["properties"]["left"] == minx and feature["properties"]["right"] == maxx and feature["properties"]["bottom"] == miny and feature["properties"]["top"] == maxy:
                # found
                if feature["properties"]["blatt_100"]:
                    return feature["properties"]["blatt_100"]
                if feature["properties"]["blatt_ostmark"]:
                    return feature["properties"]["blatt_ostmark"]
                if feature["properties"]["blatt_polen"]:
                    return feature["properties"]["blatt_polen"]
import json

from config import sheet_name_field

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

def get_ordered_bboxes_from_json(filepath, sheet_names):
    bboxes = {}

    with open(filepath, encoding="utf8") as file:
        json_data = json.load(file)

        for feature in json_data["features"]:
            if "blatt_100" in feature["properties"] and feature["properties"]["blatt_100"]:
                sheet_name = feature["properties"]["blatt_100"]
            elif "blatt_polen" in feature["properties"] and feature["properties"]["blatt_polen"]:
                sheet_name = feature["properties"]["blatt_polen"]
            elif "blatt_ostmark" in feature["properties"] and feature["properties"]["blatt_ostmark"]:
                sheet_name = feature["properties"]["blatt_ostmark"]
            elif sheet_name_field in feature["properties"] and feature["properties"][sheet_name_field]:
                sheet_name = feature["properties"][sheet_name_field]
            else:
                raise ValueError("bad format for sheets file")

            if not sheet_name in sheet_names:
                continue

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
            bboxes[sheet_name] = bbox

    ordered_bboxes = [bboxes[n] for n in sheet_names]

    return ordered_bboxes

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

def find_bbox_for_name(sheetfile, name):
    with open(sheetfile) as file:
        json_data = json.load(file)

        for feature in json_data["features"]:

            if feature["properties"]["blatt_100"] == name or feature["properties"]["blatt_ostmark"] == name or feature["properties"]["blatt_polen"] == name:
                # found
                
                minx = min([p[0] for p in feature["geometry"]["coordinates"][0]])
                maxx = max([p[0] for p in feature["geometry"]["coordinates"][0]])
                miny = min([p[1] for p in feature["geometry"]["coordinates"][0]])
                maxy = max([p[1] for p in feature["geometry"]["coordinates"][0]])

                return [minx, miny, maxx, maxy]

def find_poly_for_name(sheetfile, name):
    with open(sheetfile) as file:
        json_data = json.load(file)

        for feature in json_data["features"]:

            if feature["properties"]["blatt_100"] == name or feature["properties"]["blatt_ostmark"] == name or feature["properties"]["blatt_polen"] == name:
                # found

                return feature["geometry"]["coordinates"][0]

def get_poly_dict(sheetfile):
    return_dict = {}

    with open(sheetfile) as file:
        json_data = json.load(file)

        for feature in json_data["features"]:
            if feature["properties"]["blatt_polen"]:
                name =  feature["properties"]["blatt_polen"]
            if feature["properties"]["blatt_ostmark"]:
                name =  feature["properties"]["blatt_ostmark"]
            if feature["properties"]["blatt_100"]:
                name =  feature["properties"]["blatt_100"]
                
            return_dict[name] = feature["geometry"]["coordinates"][0]
    return return_dict

def get_dict(sheetfile, only_100=False):
    return_dict = {}

    with open(sheetfile, encoding="utf8") as file:
        json_data = json.load(file)

        for feature in json_data["features"]:

            if only_100 and (not "blatt_100" in feature["properties"] or not feature["properties"]["blatt_100"]):
                continue

            if "blatt_100" in feature["properties"] and feature["properties"]["blatt_100"]:
                name =  feature["properties"]["blatt_100"]
            elif "blatt_polen" in feature["properties"] and feature["properties"]["blatt_polen"]:
                name =  feature["properties"]["blatt_polen"]
            elif "blatt_ostmark" in feature["properties"] and feature["properties"]["blatt_ostmark"]:
                name =  feature["properties"]["blatt_ostmark"]
            elif sheet_name_field in feature["properties"] and feature["properties"][sheet_name_field]:
                name =  feature["properties"][sheet_name_field]
            else:
                print(sheet_name_field,feature["properties"])
                raise ValueError("bad format for sheets file")
                
            minx = min([p[0] for p in feature["geometry"]["coordinates"][0]])
            maxx = max([p[0] for p in feature["geometry"]["coordinates"][0]])
            miny = min([p[1] for p in feature["geometry"]["coordinates"][0]])
            maxy = max([p[1] for p in feature["geometry"]["coordinates"][0]])

            bbox = [minx, miny, maxx, maxy]

            return_dict[name] = bbox
    return return_dict
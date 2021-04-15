# import indexing
import joblib
from annoy import AnnoyIndex
import numpy as np

from config import reference_sheets_path

print("annoy: "+reference_sheets_path)
index_dict = joblib.load(reference_sheets_path)
sheets = list(index_dict.keys())
cum = np.cumsum(list(index_dict.values()))

def get_sheet_for_id(NN_id):
    pred = np.searchsorted(cum, NN_id)
    return sheets[pred]

def get_sheet_for_id_slow(index_dict, NN_id):
    desc_id = NN_id
    for key in index_dict.keys():
        num_descs = index_dict[key]
        if desc_id < num_descs:
            return key
        else:
            desc_id -= num_descs
    
    raise ValueError("sheet not found for NN id %s" % NN_id)

def get_kp_for_id(index_dict, kp_dict, NN_id):
    desc_id = NN_id
    for key in index_dict.keys():
        num_descs = len(index_dict[key])
        if desc_id < num_descs:
            kps = kp_dict[key]
            return kps[desc_id]
        else:
            desc_id -= num_descs
    
    raise ValueError("sheet/keypoint not found for NN id %s" % NN_id)

if __name__ == "__main__":
    pass
    # lookup
    # u = AnnoyIndex(64, 'euclidean')
    # u.load('test.ann') # super fast, will just mmap the file

    # q_idx = 1337

    # NN_ids = u.get_nns_by_item(q_idx, 10) # will find the n nearest neighbors

    # NN_names = [get_sheet_for_id(index_dict,i) for i in NN_ids]
    # print(get_sheet_for_id(index_dict,q_idx))
    # print(NN_ids)
    # print(NN_names)


    # v = 
    # NN_ids = u.get_nns_by_vector(v, 10) # will find the n nearest neighbors

    # NN_names = [get_sheet_for_id(index_dict,i) for i in NN_ids]
    # print(NN_names)

    u = AnnoyIndex(64, "euclidean")
    u.load('eval/index_annoydist_dot/index.ann')
    print(u)
    print(u.get_n_items())
    print(u.get_nns_by_item(0,2))
    print(get_sheet_for_id(337))
    exit()
    kp = joblib.load("sheets.clf")
    print(kp["258"])
    # print([x for x in kp.items() if x[1] != 300])
    # index = 133700
    # import time
    # iterations = 100000
    # t0 = time.time()
    # for i in range(iterations):
    #     shee = get_sheet_for_id_slow(kp,index)
    # t1 = time.time()
    # print(shee,(t1-t0))
    # t0 = time.time()
    # for i in range(iterations):
    #     pred = np.searchsorted(cum, index)
    # t1 = time.time()
    # shee = (list(kp.keys())[pred])
    # print(shee,(t1-t0))

    # print(list(kp.items())[0])
    # exit()
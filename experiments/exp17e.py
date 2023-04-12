# exp 17e: find exact time for segmentation, indexing, feature extraction

import os, shutil, glob
from datetime import datetime

restrict_hypos = 1

if __name__ == "__main__":
    sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
    images_list = "E:/data/deutsches_reich/SLUB/cut/raw/list.txt"
    annotations = "E:/data/deutsches_reich/SLUB/cut/raw/annotations.csv"

    # save old config


    data_dir = "E:/data/deutsches_reich/SLUB/cut/raw/"
    exp_dir = "E:/experiments/test1noimg/"
    out_dir = "E:/experiments/e17e/"
    os.makedirs(out_dir,exist_ok=True)

    log_files = glob.glob(f"{exp_dir}/*.log")
    segmentation_times = []
    feature_times = [] # can't tell from logs alone. feature extraction and indexing is together
    indexing_times = []
    sheet_total_times = []
    registration_times = []
    timestamp = 0
    filetime  = None
    for log_file in log_files:
        print(log_file)
        with open(log_file) as fr:
            for line in fr:
                if "Processing file" in line:
                    print(line.split("gt: ")[-1].strip())
                    old_filetime = filetime
                    filetime = line.split(" [")[0]
                    filetime = datetime.strptime(filetime,'%Y-%m-%d %H:%M:%S,%f')
                    if old_filetime:
                        delta_t = filetime-old_filetime
                        sheet_total_times.append(delta_t.total_seconds())
                # get timestamp
                old_timestamp = timestamp
                timestamp = line.split(" [")[0]
                try:
                    timestamp = datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S,%f')
                except:
                    continue
                # print(timestamp)
                # find line for segmentation
                if "segmented" in line:
                    delta_t = timestamp-old_timestamp
                    # print(delta_t.total_seconds())
                    segmentation_times.append(delta_t.total_seconds())
                # find line for features
                elif "features in query image" in line:
                    delta_t = timestamp-old_timestamp
                    # print(delta_t.total_seconds())
                    feature_times.append(delta_t.total_seconds())
                # find line for indexing
                elif "Truth at position" in line:
                    delta_t = timestamp-old_timestamp
                    # print(delta_t.total_seconds())
                    indexing_times.append(delta_t.total_seconds())
                # find line for registration
                # elif "found registration" in line:
                    # delta_t = timestamp-old_timestamp
                    # print(delta_t.total_seconds())
                    # registration_times.append(delta_t.total_seconds())
                elif "s for registration" in line:
                    time = float(line.strip().split("time: ")[-1].split(" s")[0])
                    registration_times.append(time)

    avg_segmentation_time = sum(segmentation_times)/len(segmentation_times)
    avg_feature_time = sum(feature_times)/len(feature_times)
    avg_indexing_time = sum(indexing_times)/len(indexing_times)
    avg_registration_time = 0#sum(registration_times)/len(registration_times)
    avg_sheet_total_time = sum(sheet_total_times)/len(sheet_total_times)
    print(avg_segmentation_time) 
    print(avg_feature_time)
    print(avg_indexing_time)
    print(avg_registration_time)
    print(avg_sheet_total_time)
    avg_sheet_wo_reg = 0#sum([t-r for t,r in zip(sheet_total_times,registration_times)])/len(registration_times)
    print(avg_sheet_wo_reg)

    # e8
    # 1.6963789954337891
    # ?
    # 0.669975646879756
    # 0.5022942271880821
    # 26.12707317073171
    # 27.?

    # with separate feature times 507+
    # 1.4658809523809522
    # 0.3397142857142857
    # 0.1934821428571429
    # 0.838081620689655
    # 3.9731257485029956
    # 3.0909183793103443

    # 507+ with r=30
    # 1.5313636363636363
    # 0.42918181818181816
    # 0.18581818181818177
    # 1.2790264838709677
    # 16.97974418604651
    # 15.234328354838707

    # 507+ with r=30 --noimg
    # 1.438779761904762
    # 0.3677380952380952
    # 0.22169642857142846
    # 0
    # 14.821742514970063
    # 0

    # 507+ with r=1 --noimg
    # 1.4099224242424235
    # 0.3089224242424243
    # 0.19679636363636352
    # 0
    # 2.247231796116506
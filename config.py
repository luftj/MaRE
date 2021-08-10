base_path="E:/experiments/icc_kdr500coarse/"
path_output = base_path+""#E:/experiments/fullslub/" # end with slash /
path_osm = base_path+"osm/" # end with slash /
path_logs = base_path+"logs/" # end with slash /
base_index_path = base_path+"index/"

# KDR500 uses Bonne projection
# proj_map = "+proj=bonne +lon_0=0 +lat_1=60 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
# proj_sheets = proj_map
proj_osm = "+proj=longlat +datum=WGS84 +ellps=WGS84 +no_defs" # EPSG:4326
proj_out = proj_osm
proj_sheets = proj_osm
proj_map = proj_osm
# proj_osm = proj_map

osm_url = "https://nc.hcu-hamburg.de/api/interpreter"
osm_query = """[out:json];
                (
                nwr ({{bbox}}) [water=lake]; 
                way ({{bbox}}) [natural=water] [name]; 
                way ({{bbox}}) [type=waterway] [name]; 
                way ({{bbox}}) [waterway=river] [name];
                way ({{bbox}}) [waterway=canal] [name];
                way ({{bbox}}) [natural=coastline];
                way ({{bbox}}) [waterway=riverbank];
                );
                out body;
                >;
                out skel qt;"""
                # way ({{bbox}}) [water=river] [name];
                # way ({{bbox}}) [waterway=stream] [name];
force_osm_download = False
download_timeout = (5,600) # connect timeout, read timeout
draw_ocean_polygon = True
fill_polys = True
sheet_name_field = "blatt_100"

process_image_width = 500 # image size to do all the processing in (retrieval and registration)

# opencv default resizing is linear.
# area looks good for downscaling, cubic looks good or upscaling
from cv2 import INTER_AREA, INTER_LINEAR, INTER_CUBIC
resizing_index_building = INTER_AREA
resizing_input = INTER_AREA
# resizing_template_query = INTER_LINEAR
# resizing_template_reference = INTER_AREA
resizing_index_query = INTER_AREA
resizing_register_query = INTER_AREA
resizing_register_reference = INTER_CUBIC

index_img_width_query = 500
index_n_descriptors_query = 500
index_k_nearest_neighbours = 50
index_voting_scheme = "antiprop"
index_lowes_test_ratio = None # 0.8

# the following indexing parameters require rebuilding the index
index_img_width_train = 500
index_border_train = 30
index_annoydist = "euclidean"
index_n_descriptors_train = 300
detector = kp_detector = "kaze_upright"
# possible detectors: "kaze_upright","akaze_upright","sift","surf_upright","ski_fast","cv_fast"
index_descriptor_length = 64 # depends on detector!
index_num_trees = 10

reference_sheets_path = base_index_path+"/sheets.clf"
reference_index_path = base_index_path+"index.ann"
reference_descriptors_path = base_index_path+"index.clf"
reference_descriptors_folder = base_index_path+"descriptors"
reference_keypoints_path = base_index_path+"keypoints.clf"
reference_keypoints_folder = base_index_path+"keypoints"

template_window_size = 30 # todo: is this param still in use?

#segmentation_colourbalance_percent = 5
segmentation_colourbalance_percent = 0
segmentation_blurkernel = (0,0)#(19,19)
segmentation_colourspace = "hsv" # can be ["lab","hsv"]
# HSV 
segmentation_lowerbound = (60,  15,  50)
# HSV 
segmentation_upperbound = (110, 200, 255)
# lab segmentation_lowerbound = (0,0,10)
# lab segmentation_upperbound = (255, 90, 100)#(255,70,80) #(255, 90, 80) # (255, 90, 70)
segmentation_openingkernel = (11,11)# (11,11)
segmentation_closingkernel = (11,11)

ransac_max_trials = 1000
ransac_stop_probability = 0.99
ransac_random_state = 1337 # only for profiling and validation. default: None

codebook_response_threshold = None # set to None to disable
from cv2 import NORM_INF, NORM_L1, NORM_L2, NORM_L2SQR, NORM_HAMMING, NORM_RELATIVE, NORM_MINMAX
matching_norm = NORM_L2
matching_crosscheck = True

warp_mode_retrieval = "similarity"
warp_mode_registration = "affine"

registration_mode = "both" # possible: ["ransac","ecc","both"]

registration_ecc_iterations = 500 # maximum number of ECC iterations
registration_ecc_eps = 1e-4 #threshold of the increment in the correlation coefficient between two iterations

# save disk space:
# gdal_output_options = '-a_srs "' + proj_out + '" -a_nodata 0 -of JP2OpenJPEG -co "QUALITY=5"'# -co "TILED=YES"'# -co "COMPRESS=JPEG" -co "PHOTOMETRIC=YCBCR" '
# output_file_ending = "jp2" # without dot . # for georeferenced map
# jpg2k options: https://gdal.org/drivers/raster/jp2openjpeg.html
# QUALITY defaults to 25

# save georeferencing time:
gdal_output_options = '-a_srs "' + proj_out + '" -a_nodata 0 -of GTiff --config GDAL_CACHEMAX 15000'# -co NUM_THREADS=ALL_CPUS'# -co NUM_THREADS=ALL_CPUS'
output_file_ending = "tiff"#"jp2" # without dot . # for georeferenced map

# for aligned (not georeferenced) map image
jpg_compression = None # default 95
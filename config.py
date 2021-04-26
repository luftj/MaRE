path_output = "E:/experiments/register_eval/" # end with slash /
# path_osm = "./data/osm_old/" # end with slash /
# path_osm = "E:/experiments/osm_drain_reproj/" # end with slash /
path_osm = "E:/experiments/osm_drain/" # end with slash /
path_logs = "E:/experiments/logs_evalphase_register/"#"./logs/" # end with slash /

proj_map = "+proj=longlat +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +no_defs" # Potsdam datum
proj_sheets = proj_map
proj_osm = "+proj=longlat +datum=WGS84 +ellps=WGS84 +no_defs" # EPSG:4326#
proj_out = proj_osm
# proj_osm = proj_map

osm_url = "https://nc.hcu-hamburg.de/api/interpreter"
#"http://overpass-api.de/api/interpreter"
#"https://overpass.openstreetmap.ru/api/interpreter"
#"https://overpass.osm.ch/api/interpreter"
#"http://overpass-api.de/api/interpreter"
osm_query = """[out:json];
                (nwr ({{bbox}}) [water=lake]; 
                way ({{bbox}}) [natural=water] [name]; 
                way ({{bbox}}) [type=waterway] [name]; 
                way ({{bbox}}) [waterway=river] [name];
                way ({{bbox}}) [waterway=canal] [name];
                way ({{bbox}}) [water=river];
                way ({{bbox}}) [waterway=stream] [name];
                way ({{bbox}}) [natural=coastline];
                );
                out body;
                >;
                out skel qt;"""
                # way (%s) [waterway=riverbank];
                # way (%s) [waterway=ditch];
                # way (%s) [waterway=drain];

force_osm_download = False


index_img_width_query = 500
# todo: resize interpolation method
index_n_descriptors_query = 500
index_k_nearest_neighbours = 50
index_voting_scheme = "antiprop"
index_lowes_test_ratio = None # 0.8

# the following indexing parameters require rebuilding the index
index_img_width_train = 500
# todo: resize interpolation method
index_border_train = 30
index_annoydist = "euclidean"
index_n_descriptors_train = 300
detector = kp_detector = "kaze_upright"
# possible detectors: "kaze_upright","akaze_upright","sift","surf_upright","ski_fast","cv_fast"
index_descriptor_length = 64 # depends on detector!
index_num_trees = 10

reference_sheets_path = "index/sheets.clf"
reference_index_path = "index/index.ann"
reference_descriptors_path = "index/index.clf"
reference_descriptors_folder = "index/descriptors"
reference_keypoints_path = "index/keypoints.clf"
reference_keypoints_folder = "index/keypoints"

template_window_size = 30

segmentation_colourbalance_percent = 5
segmentation_blurkernel = (19,19)
segmentation_colourspace = "lab" # can be ["lab","hsv"]
# HSV segmentation_lowerbound = (120,  0,  90)
# HSV segmentation_upperbound = (255, 255, 255)
segmentation_lowerbound = (0,0,10)
segmentation_upperbound = (255, 90, 100)#(255,70,80) #(255, 90, 80) # (255, 90, 70)
segmentation_openingkernel = (0,0)# (11,11)

ransac_max_trials = 1000
ransac_stop_probability = 0.99
ransac_random_state = 1337 # only for profiling and validation. default: None

codebook_response_threshold = 2 # maybe even 1.8 # todo: allow setting to None to disable

warp_mode_retrieval = "similarity"
warp_mode_registration = "affine"

registration_mode = "both" # possible: ["ransac","ecc","both"]

gdal_output_options = '-a_srs "' + proj_out + '" -a_nodata 0 -of JP2OpenJPEG -co "QUALITY=5"'# -co "TILED=YES"'# -co "COMPRESS=JPEG" -co "PHOTOMETRIC=YCBCR" '
# gdal_output_options = '-a_srs EPSG:4326 -a_nodata 0 -of JP2OpenJPEG -co "QUALITY=5"'# -co "TILED=YES"'# -co "COMPRESS=JPEG" -co "PHOTOMETRIC=YCBCR" '

# jpg2k options: https://gdal.org/drivers/raster/jp2openjpeg.html
# QUALITY defaults to 25
output_file_ending = "jp2" # without dot .
jpg_compression = 90 # default 95
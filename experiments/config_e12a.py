base_path = "E:/experiments/e12a/"
path_output = base_path # end with slash /
path_logs = base_path # end with slash /
base_path_index = "E:/experiments/idx_kdr100/"
path_osm = base_path_index+"/osm/" # end with slash /
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
                (
                nwr ({{bbox}}) [water=lake]; 
                nwr ({{bbox}}) [water=reservoir]; 
                way ({{bbox}}) [natural=water] [name]; 
                way ({{bbox}}) [type=waterway] [name]; 
                way ({{bbox}}) [waterway=river] [name];
                way ({{bbox}}) [waterway=canal] [name];
                way ({{bbox}}) [water=river];
                way ({{bbox}}) [waterway=stream] [name];
                way ({{bbox}}) [natural=coastline];
                way ({{bbox}}) [waterway=ditch];
                way ({{bbox}}) [waterway=drain];
                way ({{bbox}}) [waterway=riverbank];
                );
                out body;
                >;
                out skel qt;"""
                # way ({{bbox}}) [waterway=riverbank];
force_osm_download = False
download_timeout = (5,600) # connect timeout, read timeout
draw_ocean_polygon = False
line_thickness_line = {
    "waterway=river": 2,
    "natural=coastline": 0 if draw_ocean_polygon else 5,
    "default": 1
}
line_thickness_poly = {
    "natural=coastline": 3,
    "default": 3
}

def get_thickness(properties, geom_type):
    line_thickness = line_thickness_line if geom_type=="LineString" else line_thickness_poly
    for key,thickness in line_thickness.items():
        if key == "default": 
            continue
        key, value = key.split("=")
        if key in properties and properties[key] == value:
            return thickness
    else:
        return line_thickness["default"]

fill_polys = True
osm_image_size = [1000,850]
sheet_name_field = "blatt_100"
save_transform=True

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

reference_sheets_path = base_path_index+"index/sheets.clf"
reference_index_path = base_path_index+"index/index.ann"
reference_descriptors_path = base_path_index+"index/index.clf"
reference_descriptors_folder = base_path_index+"index/descriptors"
reference_keypoints_path = base_path_index+"index/keypoints.clf"
reference_keypoints_folder = base_path_index+"index/keypoints"

template_window_size = 30

segmentation_steps = [
            ("colourbalance",1),
            ("blur",9),
            # ("convert","lab"),
            # ("threshold",[(0,0,10),(250, 90, 100)]),
            ("convert","hsv"),
            ("threshold",[(80, 35, 100),(110, 170, 240)]),
            ("open",5),
            ("close",11)
            ]

ransac_max_trials = 3000
ransac_stop_probability = 0.99
ransac_random_state = 1337 # only for profiling and validation. default: None

codebook_response_threshold = None #2 # maybe even 1.8 #set to None to disable
from cv2 import NORM_INF, NORM_L1, NORM_L2, NORM_L2SQR, NORM_HAMMING, NORM_RELATIVE, NORM_MINMAX
matching_norm = NORM_L2
matching_crosscheck = True

warp_mode_retrieval = "affine" # ["similarity","affine"]
warp_mode_registration = "affine" # ["euclidean","homography","affine"]

registration_mode = "ecc" # possible: ["ransac","ecc","both"]

registration_ecc_iterations = 5000 # maximum number of ECC iterations
registration_ecc_eps = 1e-10 #threshold of the increment in the correlation coefficient between two iterations

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
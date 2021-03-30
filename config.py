path_output = "E:/experiments/perf_test/" # end with slash /
# path_osm = "./data/osm_old/" # end with slash /
path_osm = "E:/experiments/osm_streams/" # end with slash /
path_logs = "./logs_perf_test/"#"./logs/" # end with slash /

proj_map = "+proj=longlat +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +no_defs" # Potsdam datum
proj_sheets = proj_map
proj_osm = "+proj=longlat +datum=WGS84 +ellps=WGS84 +no_defs" # EPSG:4326#
proj_out = proj_osm

template_window_size = 30

warp_mode_retrieval = "similarity"
warp_mode_registration = "affine"

gdal_output_options = '-a_srs "' + proj_out + '" -a_nodata 0 -of JP2OpenJPEG -co "QUALITY=5"'# -co "TILED=YES"'# -co "COMPRESS=JPEG" -co "PHOTOMETRIC=YCBCR" '
# gdal_output_options = '-a_srs EPSG:4326 -a_nodata 0 -of JP2OpenJPEG -co "QUALITY=5"'# -co "TILED=YES"'# -co "COMPRESS=JPEG" -co "PHOTOMETRIC=YCBCR" '

# jpg2k options: https://gdal.org/drivers/raster/jp2openjpeg.html
# QUALITY defaults to 25
output_file_ending = "jp2" # without dot .
jpg_compression = 90 # default 95
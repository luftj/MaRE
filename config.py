path_output = "E:/experiments/river_register/" # end with slash /
path_osm = "./data/osm/" # end with slash /
path_logs = "./logs/" # end with slash /

gdal_output_options = '-a_srs EPSG:4326 -a_nodata 0 -of JP2OpenJPEG -co "QUALITY=5"'# -co "TILED=YES"'# -co "COMPRESS=JPEG" -co "PHOTOMETRIC=YCBCR" '
# jpg2k options: https://gdal.org/drivers/raster/jp2openjpeg.html
# QUALITY defaults to 25
output_file_ending = "jp2" # without dot .
jpg_compression = 90 # default 95
#!/bin/env python
# -*- coding: utf-8 -*- 

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import os
import sys
import string
import shutil
import math
import json
import urllib

from gibs_layer import GIBSLayer
from utils import *

from PIL import Image
from lxml import etree
from datetime import datetime, time, timedelta
from dateutil.relativedelta import relativedelta

# Test GDAL/OGR installation
# Try `conda install gdal` otherwise
try:
    from osgeo import ogr, osr, gdal
except:
    sys.exit('ERROR: cannot find GDAL/OGR modules')

default_image = "data/Blank_RGBA_512.png"

# Keep track of runtime
log_start = datetime.now()
print("generate_earth_tiles.py\nStart time: " + str(log_start) + "\n")

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser(description='Generate earth tiles')

# Begin and end date times
parser.add_argument('--time_begin', action='store', type=str, dest='date_begin',
              default="Today",
              help='The date from which to begin searching back in format YYYY-MM-DD.  Default:  Today')
parser.add_argument('--time_stop', action='store', type=str, dest='date_stop',
              default="Today-20",
              help='The date to stop searching in format YYYY-MM-DD.  Default:  Date of last check or Today-20')

# Numeric EPSG (map projection) code (i.e. 4326:geographic, 3413:arctic, 3031:antarctic)
parser.add_argument('--epsg', action='store', type=str, dest='epsg',
              default='4326',
              help='The numeric EPSG code of the map projection.  Default:  4326 (geographic)')

# Flag to request tiled world instead of whole world image
parser.add_argument('--tiled_world', action='store_true',dest='tiled_world',
              help='Flag to download the entire world as a series of tiled images.')

# Output directory
parser.add_argument('--output_dir', action='store', type=str, dest='output_dir',
              default='data',
              help='Full path of output directory.  Default:  ./data')

# Store arguments
args = parser.parse_args()

epsg = args.epsg
# tile_resolution = args.tile_resolution
tiled_world = args.tiled_world

# Parse the date information
if args.date_begin == "Today":
    date_begin = datetime.now()
else:
    date_begin = datetime.strptime(args.date_begin,"%Y-%m-%d")
if args.date_stop == "Today-20":
    date_stop = datetime.now() - timedelta(days=20)
else:
    date_stop = datetime.strptime(args.date_stop,"%Y-%m-%d")

###############################################################################
# Read files
###############################################################################

# Check valid tile resolution
# if  tile_resolution not in ["2km", "1km", "500m", "250m", "31.25m"]:
# 	print("Invalid tile_resolution.")
# 	exit()

# Create output directory if it does not exist
output_dir = args.output_dir
if output_dir[0] != '/':
    output_dir = os.getcwd() +'/' + output_dir

if os.path.exists(output_dir):
    output_dir = output_dir + "/" + epsg
    print("Outputting to " + output_dir)
else:
    exit()

if not os.path.exists(output_dir):
    print("Creating directory " + output_dir)
    os.makedirs(output_dir)

# Print out arguments
print("Map Projection: EPSG" + epsg)
print("Output directory: " + output_dir)
if args.date_stop == "Today-20":
    print("Search dates: " + datetime.strftime(date_begin,"%Y-%m-%d"))
else:
    print("Search dates: " + datetime.strftime(date_begin,"%Y-%m-%d") + " to " + datetime.strftime(date_stop,"%Y-%m-%d"))

###############################################################################
# Layer definitions (Most popular ones for MODIS Terra and VIIRS SNNP)
###############################################################################

MODIS_Terra_CorrectedReflectance_TrueColor = GIBSLayer(title="MODIS TERRA", layer_name="MODIS_Terra_CorrectedReflectance_TrueColor", epsg=epsg, format="JPEG", tilematrixset="250m", time=datetime.now())
MODIS_Terra_CorrectedReflectance_Bands367 = GIBSLayer(title="MODIS TERRA, Bands 367", layer_name="MODIS_Terra_CorrectedReflectance_Bands367", epsg=epsg, format="JPEG", tilematrixset="250m", time=datetime.now())
MODIS_Terra_Chlorophyll_A = GIBSLayer(title="MODIS Terra Chlorophyll A", layer_name="MODIS_Terra_Chlorophyll_A", epsg=epsg, format="PNG", tilematrixset="1km", time=datetime.now())
MODIS_Terra_Land_Surface_Temp_Day = GIBSLayer(title="MODIS TERRA Daytime Land Surface Temperature", layer_name="MODIS_Terra_Land_Surface_Temp_Day", epsg=epsg, format="PNG", tilematrixset="1km", time=datetime.now())
MODIS_Terra_NDVI_8Day = GIBSLayer(title="MODIS Terra NDVI 8Day", layer_name="MODIS_Terra_NDVI_8Day", epsg=epsg, format="PNG", tilematrixset="250m", time=datetime.now())

VIIRS_SNPP_CorrectedReflectance_TrueColor = GIBSLayer(title="VIIRS SNPP True Color", layer_name="VIIRS_SNPP_CorrectedReflectance_TrueColor", epsg=epsg, format="JPEG", tilematrixset="250m", time=datetime.now())
VIIRS_SNPP_DayNightBand_ENCC = GIBSLayer(title="VIIRS SNPP DayNightBand ENCC", layer_name="VIIRS_SNPP_DayNightBand_ENCC", epsg=epsg, format="PNG", tilematrixset="500m", time=datetime.now())
VIIRS_SNPP_Brightness_Temp_BandI5_Day = GIBSLayer(title="VIIRS SNPP Brightness Temp BandI5 Night", layer_name="VIIRS_SNPP_Brightness_Temp_BandI5_Day", epsg=epsg, format="PNG", tilematrixset="250m", time=datetime.now())

# No data mask
MODIS_Terra_Data_No_Data = GIBSLayer(title="MODIS TERRA Data No Data", layer_name="MODIS_Terra_Data_No_Data", epsg=epsg, format="PNG", tilematrixset="250m", time=datetime.now())

layer_dict = {
	"MODIS_Terra_CorrectedReflectance_TrueColor": MODIS_Terra_CorrectedReflectance_TrueColor,
	"MODIS_Terra_CorrectedReflectance_Bands367": MODIS_Terra_CorrectedReflectance_Bands367,
	"MODIS_Terra_Chlorophyll_A": MODIS_Terra_Chlorophyll_A,
	"MODIS_Terra_Land_Surface_Temp_Day": MODIS_Terra_Land_Surface_Temp_Day,
	"MODIS_Terra_NDVI_8Day": MODIS_Terra_NDVI_8Day,

	"VIIRS_SNPP_CorrectedReflectance_TrueColor": VIIRS_SNPP_CorrectedReflectance_TrueColor, 
	"VIIRS_SNPP_DayNightBand_ENCC": VIIRS_SNPP_DayNightBand_ENCC, 
	"VIIRS_SNPP_Brightness_Temp_BandI5_Day": VIIRS_SNPP_Brightness_Temp_BandI5_Day, 

	"MODIS_Terra_Data_No_Data": MODIS_Terra_Data_No_Data,
}

###############################################################################
# Helper functions
###############################################################################

def run_command(cmd):
    """
    Runs the provided command on the terminal.
    Arguments:
        cmd: the command to be executed.
    """
    print(' '.join(cmd))
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    process.wait()
    for output in process.stdout:
        if b"ERROR" in output:
            raise Exception(error.strip())
    for error in process.stderr:
        raise Exception(error.strip())
 
def get_bbox(x, y, num_x, num_y, epsg):
	"""
	Gets the bounding box centered at (x, y)
	Arguments: 
		x: x tile-coordinate
		y: y tile-coordinate
		epsg: Projection code
	"""
	if epsg == "4326":
		x_min = -180.0
		y_max = 90.0

		bbox_width = 180.0 / float(num_x) 
		bbox_height = 90.0 / float(num_y)

		ulx = x_min + (x * bbox_width)  # stride is same as width
		uly = y_max - (y * bbox_height)
		lrx = ulx + bbox_width
		lry = uly - bbox_height # stride is same as height
	# elif epsg == "3413" or epsg == "3031":
	#     ulx = x - 32768
	#     uly = y + 32768
	#     lrx = x + 32768
	#     lry = y - 32768
	return ulx, uly, lrx, lry

###############################################################################
# Main Loop
###############################################################################

## TODO: Loop through layers
## TODO: Loop through dates

# Set the layer we are parsing
layer = MODIS_Terra_CorrectedReflectance_TrueColor
date = "2018-06-21"

# Create a date directory
output_dir = output_dir + "/" + date 
if not os.path.exists(output_dir):
    print("Creating directory " + output_dir)
    os.makedirs(output_dir)

# Pull tiled pieces
if tiled_world:
	# Create a tiles subdirectory
	output_dir = output_dir + "/tiles" 
	if not os.path.exists(output_dir):
	    print("Creating directory " + output_dir)
	    os.makedirs(output_dir)

	piece_counter = 0
	num_x, num_y = 40, 20
	for y in range(num_y):
		for x in range(num_x):
			ulx, uly, lrx, lry = get_bbox(x, y, num_x, num_y, epsg)

			# Build the XML input file
			layer.generate_xml("twms", date)
			infile = layer.gibs_xml

			# Build name of image output file
			outfile = output_dir + "/" + layer.layer_name + "_" + str(piece_counter) + "." + layer.format_suffix

			cmd = ["gdal_translate", "-of", layer.format, "-co", "WORLDFILE=YES", "-outsize", "512", "512", "-projwin", str(ulx), str(uly), str(lrx), str(lry), infile, outfile]
			try:
			    run_command(cmd)
			except Exception as e:
				print(e)

			piece_counter += 1

# Pull the entire world!
else:
	layer.generate_xml("tms", date)
	infile = layer.gibs_xml
	infile = infile.replace("{Time}",  date)

	# Build name of output file
	outfile = output_dir + "/" + layer.layer_name + "." + layer.format_suffix

	cmd = ["gdal_translate", "-of", layer.format, "-co", "WORLDFILE=YES",  "-outsize", "51200", "25600", "-projwin", "-180", "90", "180", "-90", infile, outfile]
	try:
	    run_command(cmd)
	except Exception as e:
		print(e)
#!\\bin\\env python
# -*- coding: utf-8 -*-

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:\\\\www.apache.org\\licenses\\LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import urllib.request
import sys

from gibs_layer import GIBSLayer
from utils import *

from PIL import Image
from lxml import etree

from datetime import datetime, time, date
from dateutil.relativedelta import relativedelta

from multiprocessing.pool import ThreadPool
from time import time as timer

# Test GDAL installation
# Try `conda install gdal` otherwise
try:
    from osgeo import gdal, ogr
except:
    sys.exit('ERROR: Cannot find GDAL module')

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser(description='Download the layer tiles from the GIBS API')

# Layer name to download
parser.add_argument('--layer_name', action='store', type=str, dest='layer_name',
                    default='VIIRS_SNPP_Fires_375m_Day',
                    help='The layer name to download.  Default:  VIIRS_SNPP_Fires_375m_Day')

# Begin and end date times
parser.add_argument('--start_date', action='store', type=str, dest='start_date',
                    default=None,
                    help='The date from which to begin (inclusive) searching back in format YYYY-MM-DD.  Default:  None (uses layer start date)')
parser.add_argument('--end_date', action='store', type=str, dest='end_date',
                    default="2018-07-24",  # "Today"
                    help='The date to stop (non-inclusive) searching in format YYYY-MM-DD (or "Today").  Default:  Date of last check or Today')

# Map projection option
parser.add_argument('--epsg', action='store', type=str, dest='epsg',
                    default='4326',
                    help='The numeric EPSG code of the map projection: 4326 (geographic), 3413 (arctic), 3031 (antarctic).  Default:  4326 (geographic)')

# Tile resolution options
parser.add_argument('--tile_resolution', action='store', type=str, dest='tile_resolution',
                    default="1km",
                    help='The zoom resolution of the tiles. Must be lower than the image resolution of layer.  Default:  1km')

# Multithreading options
parser.add_argument('--num_threads', action='store', type=int, dest='num_threads',
                    default=10,
                    help='Number of concurrent threads to launch to download images.  Default:  10')

# Output directory options
parser.add_argument('--output_dir', action='store', type=str, dest='output_dir',
                    default='data',
                    help='Full path of output directory.  Default:  data')

# Store arguments
args = parser.parse_args()

# Retrieve the GIBS layer
layer_name = args.layer_name
layer = GIBSLayer.get_gibs_layer(layer_name)
if layer is None:
    print("Invalid GIBS layer name")
    exit()

# Parse the start date (inclusive) information
if args.start_date is None:
    start_date = datetime.strptime(layer.date_min, "%Y-%m-%d")
else:
    if datetime.strptime(args.start_date, "%Y-%m-%d") < datetime.strptime(layer.date_min, "%Y-%m-%d"):
        print("No layer data from {}. Will begin from {} instead".format(datetime.strptime(args.start_date, "%Y-%m-%d"),
                                                                         datetime.strptime(layer.date_min, "%Y-%m-%d")))
        start_date = datetime.strptime(layer.date_min, "%Y-%m-%d")
    else:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

# Parse the end date (non-inclusive) information
if args.end_date == "Today":
    end_date = datetime.now()
else:
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

assert start_date < end_date, "Start date is after end date!"

epsg = args.epsg

# Check valid tile resolution
tile_resolution = args.tile_resolution
if tile_resolution not in ["16km", "8km", "4km", "2km", "1km", "500m", "250m", "31.25m"]:
    print("Invalid tile_resolution.")
    exit()

num_threads = args.num_threads

# Check if output directory exists
output_dir = args.output_dir
# output_dir = os.path.join(os.getcwd(), output_dir)

if os.path.exists(output_dir):
    output_dir = os.path.join(output_dir, epsg)
else:
    print("Could not find directory " + output_dir)
    exit()

# Create EPSG code subdirectory if it does not exist (e.g. .\\data\\4326\\)
if not os.path.exists(output_dir):
    print("Creating directory " + output_dir)
    os.makedirs(output_dir)

# Print out arguments
print("Downloading layer: " + layer_name)
print("Dates Range: " + datetime.strftime(start_date, "%Y-%m-%d") + " (inclusive) up to " + datetime.strftime(end_date,
                                                                                                              "%Y-%m-%d") + " (non-inclusive)")
print("Map Projection: EPSG" + epsg)
print("Output directory: " + output_dir)

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
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    for output in process.stdout:
        if b"ERROR" in output:
            raise Exception(error.strip())
    for error in process.stderr:
        raise Exception(error.strip())


def get_tiled_grid_dim(tile_resolution):
    tile_grid_dict = {
        "16km": (8, 4),
        "8km": (16, 8),
        "4km": (32, 16),
        "2km": (64, 32),
        "1km": (128, 64),
        "500m": (256, 128),
        "250m": (512, 256),
    }

    if tile_resolution in tile_grid_dict:
        return tile_grid_dict[tile_resolution]

    return None


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
        lry = uly - bbox_height  # stride is same as height
    # elif epsg == "3413" or epsg == "3031":
    #     ulx = x - 32768
    #     uly = y + 32768
    #     lrx = x + 32768
    #     lry = y - 32768
    return ulx, uly, lrx, lry


###############################################################################
# Main Loop
###############################################################################

# List of commands to pass to the thread pool
commands = []
url_commands = []

# Base directory containing dates
output_root_dir = output_dir

# Loop through dates
for single_date in daterange(start_date, end_date):
    datestring = single_date.strftime("%Y-%m-%d")

    # Create a date directory
    output_dir = os.path.join(output_root_dir, datestring)
    if not os.path.exists(output_dir):
        print("Creating directory " + output_dir)
        os.makedirs(output_dir)

    # Create a tiles subdirectory
    output_dir = os.path.join(output_dir, "tiles")
    if not os.path.exists(output_dir):
        print("Creating directory " + output_dir)
        os.makedirs(output_dir)

    num_x, num_y = get_tiled_grid_dim(tile_resolution)
    for y in range(num_y):
        for x in range(num_x):
            ulx, uly, lrx, lry = get_bbox(x, y, num_x, num_y, epsg)

            tile_url = "https://gibs-a.earthdata.nasa.gov/wms/wms.php?TIME=${DATE}&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image/${FORMAT}&TRANSPARENT=true&LAYERS=${LAYER}&WIDTH=512&HEIGHT=512&SRS=EPSG:4326&STYLES=&BBOX=${ulx},${lry},${lrx},${uly}"
            tile_url = "https://gibs-c.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi?TIME={DATE}T00:00:00Z&layer=VIIRS_SNPP_CorrectedReflectance_TrueColor&style=default&tilematrixset=250m&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image%2F{FORMAT}&TileMatrix=0&TileCol=0&TileRow=0"
            
            # Fill in layer and date information
            # VIIRS_SNPP_Fires_375m_Day
            tile_url = tile_url.replace("${DATE}", datestring)
            tile_url = tile_url.replace("${FORMAT}", layer.format)
            if layer.layer_name == "VIIRS_SNPP_Fires_375m_Day" or layer.layer_name == "VIIRS_SNPP_Fires_375m_Night":
                tile_url = tile_url.replace("${LAYER}", "VIIRS_SNPP_Fires_375m_Day,VIIRS_SNPP_Fires_375m_Night")
            else:
                tile_url = tile_url.replace("${LAYER}", layer.layer_name)

            # Fill in the bounding box parameters
            tile_url = tile_url.replace("${ulx}", str(ulx))
            tile_url = tile_url.replace("${lry}", str(lry))
            tile_url = tile_url.replace("${uly}", str(uly))
            tile_url = tile_url.replace("${lrx}", str(lrx))

            # print(tile_url)

            # Build name of image output file
            grid_coord = "Tile_" + str(x) + "-" + str(y)
            if layer.layer_name == "VIIRS_SNPP_Fires_375m_Day" or layer.layer_name == "VIIRS_SNPP_Fires_375m_Night":
                outfile = os.path.join(output_dir, "VIIRS_SNPP_Fires_375m" + "_" + grid_coord + "." + layer.format_suffix)
            else:
                outfile = os.path.join(output_dir, layer.layer_name + "_" + grid_coord + "." + layer.format_suffix)
            url_commands.append((tile_url, outfile))

            # urllib.request.urlretrieve(tile_url, outfile)

            # cmd = ' '.join(cmd_list)
            # commands.append(cmd)
            # print(cmd)

            # try:
            #     run_command(cmd)
            # except Exception as e:
            # 	print(e)


###############################################################################
# Launch ThreadPool to download tiles
###############################################################################

num_commands = len(url_commands)
print("Preparing to download {} images...".format(str(num_commands)))


def fetch_url(url_cmd):
    url, outfile = url_cmd
    try:
        filename, headers = urllib.request.urlretrieve(url, outfile)
        return url, None
    except Exception as e:
        return url, e


# Start the timer
start = timer()

# Launch the commands using a thread pool
pool = ThreadPool(num_threads)  # maximum of num_threads concurrent commands at a time
results = pool.imap_unordered(fetch_url, url_commands)
for url, error in results:
    if error is None:
        print("%r fetched in %ss" % (url, timer() - start))
    else:
        print("error fetching %r: %s" % (url, error))

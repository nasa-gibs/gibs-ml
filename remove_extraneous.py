# Copyright 2018 California Institute of Technology.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os, re
import random
import numpy as np
from shutil import copyfile

from PIL import Image
from tqdm import tqdm

from datetime import datetime, time, timedelta, date
from dateutil.relativedelta import relativedelta
from utils import daterange

from gibs_layer import GIBSLayer

###############################################################################
# Constants
###############################################################################
layer_name = 'MODIS_Terra_NDVI_8Day'
layer = GIBSLayer.get_gibs_layer(layer_name)

# Start and end dates for the labels
start_date = datetime.strptime("2015-11-24","%Y-%m-%d") 
end_date = datetime.strptime(layer.date_min,"%Y-%m-%d") # data only begins from here! 
assert start_date < end_date, "Start date is after end date!"

# train-val-test split (60-20-20)
TRAIN_CUTOFF = 0.60
VAL_CUTOFF = 0.80

# directory pointing to date subdirectories
data_dir = 'data/4326'

###############################################################################
# File checking
###############################################################################
assert os.path.isdir(data_dir), "Couldn't find the dataset at {}".format(data_dir)

# Loop through dates
for single_date in daterange(start_date, end_date):
	datestring = single_date.strftime("%Y-%m-%d")
	print(datestring)
	path_directory = os.path.join(data_dir, datestring)
	for f in os.listdir(path_directory):
		file_pattern = layer_name + ".*"
		if re.search(file_pattern, f):
			print("-- Removing file: {}".format(f))
			file_path = os.path.join(path_directory, f)
			os.remove(file_path)
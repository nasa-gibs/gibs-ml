import sys
import os
import random
import numpy as np
from shutil import copyfile

from PIL import Image
from tqdm import tqdm

from datetime import datetime, time, timedelta, date
from dateutil.relativedelta import relativedelta
from utils import daterange

###############################################################################
# Constants
###############################################################################
layer_name = 'VIIRS_SNPP_CorrectedReflectance_TrueColor'

# Start and end dates for the labels
start_date = date(2015, 11, 24)
end_date = date(2018, 6, 27)

# train-val-test split (60-20-20)
TRAIN_CUTOFF = 0.60
VAL_CUTOFF = 0.80

# directory pointing to date subdirectories
data_dir = 'data/4326'
labels_file = os.path.join(data_dir, layer_name + ".txt")
split_labels_file = os.path.join(data_dir, "split_" + layer_name + ".txt")

###############################################################################
# File checking
###############################################################################
assert os.path.isdir(data_dir), "Couldn't find the dataset at {}".format(data_dir)

# Generate labels file if non-existent
if not os.path.exists(labels_file):
    print("Making labels file {}".format(labels_file))
    with open(labels_file, "w") as f:
        # Loop through dates
        for single_date in daterange(start_date, end_date):
            datestring = single_date.strftime("%Y-%m-%d")
            f.write(datestring + " \n")

###############################################################################
# Add split to the labels file
###############################################################################

# Count the number of examples in the dataset
N = sum(1 for line in open(labels_file))
print("There are {} total examples in the dataset".format(N))

# Split the dataset into train-val-test
first_split = int(TRAIN_CUTOFF * N)
second_split = int(VAL_CUTOFF * N)

split_names = [""] * N
for idx in range(N):
    if idx < first_split:
        split_names[idx] = "train"
    elif idx < second_split:
        split_names[idx] = "val"
    else:
        split_names[idx] = "test"

# # Make sure to always shuffle with a fixed seed so that the split is reproducible
random.seed(999)
split_names.sort()
random.shuffle(split_names)

with open(labels_file, "r") as f:
    with open(split_labels_file, "w") as split_f:
        for idx, line in enumerate(f):
            split_f.write(split_names[idx] + " " + line)

print("Done updating labels file")

# # Pre-process train and validation sets
# for split in ['train', 'val']:
#     output_dir_split = os.path.join(output_directory, '{}'.format(split))
#     if not os.path.exists(output_dir_split):
#         os.mkdir(output_dir_split)
#     else:
#         print("Warning: dir {} already exists".format(output_dir_split))

#     print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
#     for filename in tqdm(filenames[split]):
#         img_name = os.path.split(filename)[-1]
#         copyfile(filename, os.path.join(output_dir_split, img_name))
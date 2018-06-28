import sys
import os
import random
import numpy as np
from shutil import copyfile

from PIL import Image
from tqdm import tqdm

# training-val-dev split
TRAIN_CUTOFF = 0.70

# directory pointing to all .jpg files
input_directory = 'data/TINYSAT'

# experiment directories
output_directory = 'data'

if __name__ == '__main__':

    assert os.path.isdir(input_directory), "Couldn't find the dataset at {}".format(input_directory)

    # Get the filenames in the input directory
    filenames = os.listdir(input_directory)
    filenames = [os.path.join(input_directory, f) for f in filenames if f.endswith('.jpg')]

    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(999)
    filenames.sort()
    random.shuffle(filenames)

    # Split the image into 70% train and 30% val
    first_split = int(TRAIN_CUTOFF * len(filenames))
    train_filenames = filenames[:first_split]
    val_filenames = filenames[first_split:]

    filenames = {'train': train_filenames,
                 'val': val_filenames}

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    else:
        print("Warning: output dir {} already exists".format(output_directory))

    # Pre-process train and validation sets
    for split in ['train', 'val']:
        output_dir_split = os.path.join(output_directory, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            img_name = os.path.split(filename)[-1]
            copyfile(filename, os.path.join(output_dir_split, img_name))

    print("Done building dataset")

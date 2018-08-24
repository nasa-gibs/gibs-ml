import random
import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import dill as pickle

import cv2

import scipy
from scipy import ndimage

class SatelliteDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """    
    def __init__(self, params, data_dir, layer_name, label_split_file, split, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            label_split_file: (string) filename containing the dataset label and split metadata
            split (string): 'train', 'val', or 'test' depending on which data is required
            transform: (torchvision.transforms) transformation to apply on image
        """
        def parse_labels(label_arr):
            """    
            None = Normal
            1 = missing data (expected)
            2 = missing data (unexpected)
            3 = miscoloration
            4 = edge warping
            5 = eclipse (missing data)
            6 = eclipse (only miscoloration)
            """
            label = 0
            if len(label_arr) == 0:
                label = 0
            else:
                if 1 in label_arr or 2 in label_arr: # or 5 in label_arr:
                    label = 1
                else:
                    label = 0
            return label

        def load_layer_split():
            filenames = []
            labels = []

            # Read in the file line by line
            with open(label_split_file) as f:
                file_lines = f.read().splitlines()
                num_total_img = len(file_lines)
                for line in file_lines:
                    if split == 'new_test':
                        datestring = line
                        filenames.append(os.path.join(data_dir, datestring, layer_name + ".jpg"))
                        labels.append(0)
                    else:
                        line_list = line.split()  
                        img_split = line_list[0]
                        if img_split == split:
                            datestring = line_list[1]
                            label_arr = [int(item) for item in line_list[2:]]

                            # Construct and resize the image
                            filenames.append(os.path.join(data_dir, datestring, layer_name + ".jpg"))
                            labels.append(parse_labels(label_arr))     
            return filenames, labels

        self.filenames, self.labels = load_layer_split()
        self.transform = transform
        self.img_binarize = params.img_binarize

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        
        # Grayscale and black and white the image!
        if self.img_binarize:
            image = np.asarray(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)            
            image = Image.fromarray(image)

        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloader(split, transformer, data_dir, layer_name, label_split_file, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        split: (string) has one of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        label_split_file: (string) filename containing the dataset label and split metadata
        params: (Params) hyperparameters

    Returns:
        data: DataLoader object for split
    """    

    dataloader = None
        
    if split == 'train':
        # Oversample the training set to deal with class imbalance
        # class_sample_count = [495, 72] # training dataset has 10 class-0 samples, 1 class-1 samples, etc.
        # num_samples = sum(class_sample_count)
        # weights = 1 / torch.Tensor(class_sample_count)
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples)
        # Must set 'shuffle' to False
        dataloader = DataLoader(SatelliteDataset(params, data_dir, layer_name, label_split_file, split, transformer), batch_size=params.batch_size, sampler=None, shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)
    elif split == 'val' or split == 'test':
        dataloader = DataLoader(SatelliteDataset(params, data_dir, layer_name, label_split_file, split, transformer), batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)
    elif split =='new_test':
        dataloader = DataLoader(SatelliteDataset(params, data_dir, layer_name, label_split_file, split, transformer), batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    return dataloader

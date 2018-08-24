import random
import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import dill as pickle
import torchvision.transforms as transforms

import cv2

import scipy
from scipy import ndimage

class ToLabel(object):
    def __call__(self, inputs):
        tensors = []
        for i in inputs:
            tensors.append(torch.from_numpy(np.array(i)).long())
        return tensors


class ReLabel(object):
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        # assert isinstance(input, torch.LongTensor), 'tensor needs to be LongTensor'
        for i in inputs:
            i[i == self.olabel] = self.nlabel
        return inputs

class SatelliteDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """    
    def __init__(self, params, data_dir, layer_name, missing_data_name, label_split_file, split, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            label_split_file: (string) filename containing the dataset label and split metadata
            split (string): 'train', 'val', or 'test' depending on which data is required
            transform: (torchvision.transforms) transformation to apply on image
        """
        def parse_img_labels(label_arr):
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
            
            # Normal image!
            if len(label_arr) == 0:
                return label
            
            if 1 in label_arr or 2 in label_arr:
                label = 1

            return label

        def load_layer_split():
            img_filenames = []
            label_filenames = []

            # Read in the file line by line
            with open(label_split_file) as f:
                file_lines = f.read().splitlines()
                for line in file_lines:
                    line_list = line.split()  
                    img_split = line_list[0]
                    if img_split == split:
                        datestring = line_list[1]

                        # Skip normal images
                        img_label = parse_img_labels([int(item) for item in line_list[2:]])
                        if img_label == 0:
                            continue

                        img_filenames.append(os.path.join(data_dir, datestring, layer_name + ".jpg"))
                        label_filenames.append(os.path.join(data_dir, datestring, missing_data_name + ".png"))     
            return img_filenames, label_filenames

        self.img_filenames, self.label_filenames = load_layer_split()
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.img_filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.img_filenames[idx]).convert('RGB')  # PIL image
        image = self.transform(image)

        target_transform = transforms.Compose([
            ToLabel(),
            ReLabel(255, 2),
        ])

        label = Image.open(self.label_filenames[idx])
        label = np.asarray(label)

        # 0-1 Binarize the labels
        label = np.where(label > 1, 1, 1)

        return image, target_transform(label)

def fetch_dataloader(split, transformer, data_dir, layer_name, missing_data_name, label_split_file, params):
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
        dataloader = DataLoader(SatelliteDataset(params, data_dir, layer_name, missing_data_name, label_split_file, split, transformer), batch_size=params.batch_size, sampler=None, shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)
    elif split == 'val' or split == 'test':
        dataloader = DataLoader(SatelliteDataset(params, data_dir, layer_name, missing_data_name, label_split_file, split, transformer), batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)
    elif split =='new_test':
        dataloader = DataLoader(SatelliteDataset(params, data_dir, layer_name, missing_data_name, label_split_file, split, transformer), batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)
    return dataloader
